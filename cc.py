import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer

P = "\033[43m"
E = "\033[0m"


## 데이터 전처리: Common Voice와 유사하게, 각 항목에서 waveform, sample_rate, transcript 반환
def preprocess(batch):
    # print(batch.keys())
    waveform, sample_rate = torchaudio.load(batch["path"])
    max_length = sample_rate * 6  # 최대 6초
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]
    transcript = batch["sentence"]
    return {"waveform": waveform, "sample_rate": sample_rate, "transcript": transcript}

# ===============================================================
# 1. 데이터셋 정의 (Fine-tuning용) - Llama 3 토크나이저 사용
# ===============================================================
class SpeechDataset(Dataset):
    """
    전처리된 데이터 항목(딕셔너리)을 입력받는 데이터셋.
    각 항목은 {"waveform", "sample_rate", "transcript"} 형태이며,
    전사는 Llama 3 토크나이저를 사용하여 토큰 인덱스로 변환됩니다.
    """
    def __init__(self, preprocessed_data, tokenizer, max_length=300):
        self.data_list = preprocessed_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        waveform = item["waveform"]
        transcript = item["transcript"]
        if waveform.dim() > 1 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        # Llama 3 토크나이저를 사용하여 전사를 토큰화 (special token 자동 추가)
        tokens = self.tokenizer(
            transcript,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        return waveform, tokens

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    waveforms_padded = pad_sequence(waveforms, batch_first=True)
    # transcripts 은 리스트로 내부 tensor는 각각 길이가 다름
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)
    return waveforms_padded, transcripts_padded

# ===============================================================
# 2. Whisper 기반 Fine-tuning 모델: Encoder (freeze) + Decoder
# ===============================================================
from w2v2 import Wav2Vec2Model

class WhisperFineTuneModel(nn.Module):
    def __init__(self, pretrained_encoder:Wav2Vec2Model, vocab_size, decoder_embed_dim=512, num_decoder_layers=6,
                 num_heads=8, ff_hidden_dim=2048, dropout=0.1, max_target_len=300):
        """
        pretrained_encoder: 사전학습된 wav2vec2 모델 (encoder 부분)
        vocab_size: 디코더의 출력 vocabulary 크기
        decoder_embed_dim: 디코더 임베딩 차원
        num_decoder_layers: 디코더 층 수
        num_heads: 디코더 멀티헤드 어텐션 헤드 수
        ff_hidden_dim: 디코더 피드포워드 네트워크 hidden 차원
        dropout: dropout 확률
        max_target_len: 최대 디코더 입력 길이 (positional embedding을 위한)
        """
        super().__init__()
        self.encoder = pretrained_encoder
        # encoder 파라미터 freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # encoder 출력 차원과 decoder 임베딩 차원이 다르면 projection 추가
        self.encoder_proj = nn.Linear(pretrained_encoder.encoder_embed_dim, decoder_embed_dim)
        
        self.token_embedding = nn.Embedding(vocab_size, decoder_embed_dim)
        self.pos_embedding = nn.Embedding(max_target_len, decoder_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=num_heads,
                                                    dim_feedforward=ff_hidden_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(decoder_embed_dim, vocab_size)
    
    def forward(self, source, target):
        """
        Args:
            source (Tensor): Raw audio waveform, shape (B, T_source)
            target (Tensor): Target token sequence with special tokens, shape (B, T_target)
        Returns:
            logits: (B, T_target, vocab_size)
        """
        # Encoder: 특징 추출 및 decoder 차원으로 프로젝션
        encoder_feats = self.encoder.forward_features(source)   # (B, T_enc, C_enc)
        encoder_feats = self.encoder_proj(encoder_feats)          # (B, T_enc, d_model)
        memory = encoder_feats.transpose(0, 1)  # Transformer expects (T_enc, B, d_model)
        
        # Decoder 입력: 토큰 임베딩 + positional 임베딩
        tgt_emb = self.token_embedding(target)  # (B, T_target, d_model)
        positions = torch.arange(0, target.size(1), device=target.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        tgt_emb = tgt_emb + pos_emb
        tgt_emb = tgt_emb.transpose(0, 1)  # (T_target, B, d_model)
        
        # 미래 토큰 참조를 막기 위한 subsequent mask 생성
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target.size(1)).to(target.device)
        
        decoder_output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # (T_target, B, d_model)
        decoder_output = decoder_output.transpose(0, 1)  # (B, T_target, d_model)
        logits = self.output_projection(decoder_output)   # (B, T_target, vocab_size)
        return logits

# ===============================================================
# 3. 사전학습 모델 로드 함수
# ===============================================================
def load_pretrained_model(checkpoint_path, device):
    model = Wav2Vec2Model()  # 사전학습 시 사용한 하이퍼파라미터로 초기화
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
# ===============================================================
# 4. Fine-tuning 학습 루프
# ===============================================================
def fine_tune(checkpoint_path, preprocessed_data, num_epochs=10, lr=1e-4, batch_size=4, max_target_len=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # preprocessed_data = [preprocess(item) for item in raw_data_list]
    
    # Llama 3 토크나이저 초기화 (실제 모델 식별자로 교체하세요)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    vocab_size = tokenizer.vocab_size

    # 사전학습된 encoder 로드
    pretrained_encoder = load_pretrained_model(checkpoint_path, device)
    
    # Whisper 기반 Fine-tuning 모델 (encoder freeze + decoder)
    model = WhisperFineTuneModel(pretrained_encoder, vocab_size+20, decoder_embed_dim=512,
                                 num_decoder_layers=6, num_heads=8, ff_hidden_dim=2048,
                                 dropout=0.1, max_target_len=max_target_len).to(device)
    
    dataset = SpeechDataset(preprocessed_data, tokenizer, max_length=max_target_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Cross entropy loss (pad token을 무시)
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for waveforms, targets in dataloader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            # teacher forcing: 디코더 입력은 target의 [:-1] 사용, loss는 target의 [1:]와 비교
            logits = model(waveforms, targets[:, :-1])
            loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ===============================================================
# 5. 메인 실행 예시: Hugging Face의 Common Voice (ko) 데이터셋 사용
# ===============================================================
if __name__ == "__main__":
    checkpoint_path = "pretrained_model.pth"    
    # Common Voice 한국어 train split 불러오기
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "ko", split="train")
    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch", columns=["waveform", "sample_rate", "transcript"])
    raw_data_list = dataset
    print(P + f"{dataset}" + E)
    print(raw_data_list)
    fine_tune(checkpoint_path, raw_data_list, num_epochs=10, lr=1e-4, batch_size=2)
