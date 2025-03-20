import torch
import torchaudio
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm  # 진행 상태 모니터링을 위한 tqdm
from w2v2mask import Wav2Vec2Model, ContrastiveLossModule
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\033[35mUsing device: {device}\033[0m")

dataset = load_dataset("ricecake/Genshin_Impact_RaidenShogun_Voice_korean", split="train")


def preprocess(batch):
    audio = batch["audio"]
    waveform, sample_rate = torchaudio.load(audio["path"])
    max_length = sample_rate * 6  # 3초에 해당하는 최대 샘플 수
    # waveform의 shape가 (채널, 시간)이라고 가정합니다.
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]  # 3초 초과 시 잘라냄
    return {"waveform": waveform, "sample_rate": sample_rate}


dataset = dataset.map(preprocess)
dataset.set_format(type="torch", columns=["waveform", "sample_rate"])

def data_collator(batch):
    waveforms = [item["waveform"].squeeze(0) for item in batch]
    waveforms = pad_sequence(waveforms, batch_first=True)
    return {"waveforms": waveforms}

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

model = Wav2Vec2Model(
    encoder_num_layers=6,
    encoder_dropout=0.3,
    quantizer_dropout=0.3,
    encoder_ff_dropout=0.4,
    feature_extractor_dropout=0.2
).to(device)
contrastive_loss_fn = ContrastiveLossModule()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        waveforms = batch["waveforms"].to(device)
        optimizer.zero_grad()

        encoded_features, quantized_features, _, negatives, mask = model(
            waveforms, mask=True, return_quantized=True, return_negatives=True
        )

        loss = contrastive_loss_fn.compute_contrastive_loss(
            encoded_features, quantized_features, mask, num_negatives=100, temperature=0.1
        )

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.8f}")