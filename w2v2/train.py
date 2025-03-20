import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
from haha import *  # wav2vec2_model, Wav2Vec2PreTrainer, base, pretrainer_config 등 포함

# 시드 고정
torch.manual_seed(0)
np.random.seed(0)

# 하이퍼파라미터 설정
B = 12  # 배치 사이즈
epochs = 10
learning_rate = 1e-4

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\033[35mUsing device: {device}\033[0m")

# 데이터 전처리 함수 (torchaudio 활용)
def preprocess(batch):
    waveform, sample_rate = torchaudio.load(batch["audio"]["path"])
    max_length = sample_rate * 6  # 최대 6초 길이
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]  # 최대 길이 초과 시 자르기
    return {"waveform": waveform, "sample_rate": sample_rate}

# 데이터셋 로드 및 전처리
print("Loading dataset...")
raw_dataset = load_dataset("ricecake/Genshin_Impact_RaidenShogun_Voice_korean", split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print("Preprocessing train dataset...")
train_dataset = train_dataset.map(preprocess)
print("Preprocessing validation dataset...")
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format(type="torch", columns=["waveform", "sample_rate"])
val_dataset.set_format(type="torch", columns=["waveform", "sample_rate"])

# collate 함수: padding 처리 및 실제 길이 정보 반환
def data_collator(batch):
    waveforms = [item["waveform"].squeeze(0) for item in batch]
    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])
    waveforms = pad_sequence(waveforms, batch_first=True)
    return {"waveforms": waveforms, "lengths": lengths}

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

# validation 함수: 모델 평가 수행
def validate_model(network, pretrainer, val_loader, pretrainer_cfg):
    network.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            x = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                c, q, q_dist, mti = pretrainer(x, lengths)
                cl = pretrainer.compute_contrastive_loss(
                    c, q, mti,
                    num_negatives=pretrainer_cfg["num_negatives"],
                    temperature=pretrainer_cfg["contrastive_loss_temperature"]
                )
                dl = pretrainer.compute_diversity_loss(
                    q_dist, mti, pretrainer.quantizer.num_vars * pretrainer.quantizer.num_groups
                )
                loss = cl + pretrainer_cfg["diversity_loss_weight"] * dl
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# 학습 함수
def train_wav2vec2():
    # 모델 및 optimizer 설정
    wav2vec2_cfg = base["base"]
    pretrainer_cfg = pretrainer_config["base"]
    
    network = wav2vec2_model(**wav2vec2_cfg).to(device)
    pretrainer = Wav2Vec2PreTrainer(network, **pretrainer_cfg).to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 학습 단계
        network.train()
        train_loss = 0.0
        train_cl_loss = 0.0
        train_dl_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            x = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                c, q, q_dist, mti = pretrainer(x, lengths)
                cl = pretrainer.compute_contrastive_loss(
                    c, q, mti,
                    num_negatives=pretrainer_cfg["num_negatives"],
                    temperature=pretrainer_cfg["contrastive_loss_temperature"]
                )
                dl = pretrainer.compute_diversity_loss(
                    q_dist, mti, pretrainer.quantizer.num_vars * pretrainer.quantizer.num_groups
                )
                loss = cl + pretrainer_cfg["diversity_loss_weight"] * dl
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_cl_loss += cl.item()
            train_dl_loss += dl.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cl = train_cl_loss / len(train_loader)
        avg_train_dl = train_dl_loss / len(train_loader)
        
        # Validation 단계 호출
        avg_val_loss = validate_model(network, pretrainer, val_loader, pretrainer_cfg)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} (CL: {avg_train_cl:.4f}, DL: {avg_train_dl:.4f}) | Val Loss: {avg_val_loss:.4f}")
        
        # 최고의 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(network.state_dict(), "wav2vec2_pretrained.pth")
            print(f"Model checkpoint saved (Val Loss: {avg_val_loss:.4f})")
    
    print("Training completed.")
    return network

if __name__ == "__main__":
    print("Starting model training...")
    trained_model = train_wav2vec2()
    print("Model weights saved successfully.")
