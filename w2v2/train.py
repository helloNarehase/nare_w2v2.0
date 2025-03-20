from haha import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
from tqdm import tqdm

# 시드 고정
torch.manual_seed(0)
np.random.seed(0)

# 하이퍼파라미터 설정
B = 12  # 배치 사이즈
T = int(16000 * 1)  # 1초 길이 (16kHz 샘플링)
epochs = 10
learning_rate = 1e-4

# 데이터셋 처리 클래스 정의
class AudioDataset(Dataset):
	def __init__(self, dataset_path, split="train"):
		"""
		Args:
			dataset_path: 전처리된 데이터셋이 저장된 경로
			split: 'train', 'val', 'test' 중 하나
		"""
		self.data_path = os.path.join(dataset_path, split)
		self.file_list = [f for f in os.listdir(self.data_path) if f.endswith('.pt')]
		
	def __len__(self):
		return len(self.file_list)
	
	def __getitem__(self, idx):
		file_path = os.path.join(self.data_path, self.file_list[idx])
		data = torch.load(file_path)
		return data['audio'], data['length']


# 데이터셋 전처리 및 저장 함수
def preprocess_and_save_dataset(output_dir, max_samples=None):
	"""
	데이터셋을 전처리하고 디스크에 저장
	
	Args:
		output_dir: 전처리된 데이터를 저장할 디렉토리
		max_samples: 처리할 최대 샘플 수 (None이면 전체)
	"""
	# 출력 디렉토리 생성
	os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
	os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
	
	# 데이터셋 로드
	print("Loading dataset...")
	dataset = load_dataset("joujiboi/japanese-anime-speech", split="train")
	
	if max_samples is not None:
		dataset = dataset.select(range(min(max_samples, len(dataset))))
	
	# 학습/검증 세트 분할 (90/10)
	dataset = dataset.train_test_split(test_size=0.1)
	train_dataset = dataset['train']
	val_dataset = dataset['test']
	
	# 학습 데이터 전처리 및 저장
	print(f"Preprocessing and saving training data ({len(train_dataset)} samples)...")
	preprocess_and_save_split(train_dataset, os.path.join(output_dir, 'train'))
	
	# 검증 데이터 전처리 및 저장
	print(f"Preprocessing and saving validation data ({len(val_dataset)} samples)...")
	preprocess_and_save_split(val_dataset, os.path.join(output_dir, 'val'))


def preprocess_and_save_split(dataset, output_dir):
	"""한 분할 데이터셋을 전처리하고 저장"""
	for i, idx in enumerate(tqdm(range(len(dataset)))):
		try:
			sample = dataset[idx]
		except:
			continue
		audio_array = sample["audio"]["array"]
		sampling_rate = sample["audio"]["sampling_rate"]
		
		# 샘플링 레이트가 16kHz가 아니면 리샘플링 (실제 구현 필요)
		if sampling_rate != 16000:
			# 리샘플링 로직 추가 필요
			# 예: audio_array = resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
			pass
		
		# 길이 조정 (고정 길이로 자르거나 패딩)
		if len(audio_array) >= T:
			audio_array = audio_array[:T]
			length = T
		else:
			length = len(audio_array)
			audio_array = np.pad(audio_array, (0, T - length), mode="constant")
		
		# 텐서로 변환
		audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
		length_tensor = torch.tensor(length, dtype=torch.int64)
		
		# 저장
		torch.save({
			'audio': audio_tensor,
			'length': length_tensor,
		}, os.path.join(output_dir, f"sample_{i:06d}.pt"))


# 모델 학습 함수
def train_wav2vec2(data_dir, model_save_path="wav2vec2_pretrained.pth"):
	
	# 데이터로더 설정
	train_dataset = AudioDataset(data_dir, split='train')
	val_dataset = AudioDataset(data_dir, split='val')
	
	train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, num_workers=4)
	
	# 모델 및 optimizer 설정
	wav2vec2_cfg = base["base"]
	pretrainer_cfg = pretrainer_config["base"]
	
	network = wav2vec2_model(**wav2vec2_cfg)
	pretrainer = Wav2Vec2PreTrainer(network, **pretrainer_cfg)
	pretrainer.train()
	
	optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
	scaler = torch.amp.GradScaler()
	
	# 학습 루프
	best_val_loss = float('inf')
	
	for epoch in range(epochs):
		# 학습 단계
		network.train()
		train_loss = 0.0
		train_cl_loss = 0.0
		train_dl_loss = 0.0
		
		for x, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
			optimizer.zero_grad()
			
			with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
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
		
		# 검증 단계
		network.eval()
		val_loss = 0.0
		
		with torch.no_grad():
			for x, lengths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
				with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
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
				
				val_loss += loss.item()
		
		avg_val_loss = val_loss / len(val_loader)
		
		print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} (CL: {avg_train_cl:.4f}, DL: {avg_train_dl:.4f}) | Val Loss: {avg_val_loss:.4f}")
		
		# 최고 성능 모델 저장
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(network.state_dict(), model_save_path)
			print(f"Model checkpoint saved to {model_save_path} (Val Loss: {avg_val_loss:.4f})")
	
	print("Training completed.")
	return network


if __name__ == "__main__":
	# 데이터 전처리 및 저장 (처음 한 번만 실행)
	data_dir = "./preprocessed_audio_data"
	if not os.path.exists(data_dir):
		print("Preprocessing dataset...")
		preprocess_and_save_dataset(data_dir, max_samples=20000)
	else:
		print(f"Using preprocessed data from {data_dir}")
	
	# 모델 학습
	print("Starting model training...")
	trained_model = train_wav2vec2(data_dir)
	
	print("Model weights saved successfully.")