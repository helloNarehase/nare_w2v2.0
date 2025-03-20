import torch
from torch import nn, Tensor
from typing import Optional
from torch.nn import functional as F

from Attention import Attention, FeedForward, TransformerBlock
    
class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings: int):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_pad_remove > 0:
            x = x[:, :, : -self.num_pad_remove]
        return x


class ConvolutionalPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = Wav2Vec2SamePadLayer(kernel_size) # num_conv_pos_embeddings -> kernel_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        포지셔널 임베딩을 입력 hidden states에 적용합니다.

        Args:
            hidden_states (torch.Tensor): shape (batch_size, sequence_length, hidden_size)를 가진 입력 텐서.

        Returns:
            torch.Tensor: 포지셔널 임베딩이 추가된 텐서.
        """
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Transformer(nn.Module):
    def __init__(
        self,
		pos_conv_embed: ConvolutionalPositionalEmbedding,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        ff_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embedding_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                ff_dropout=ff_dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def post_embed(self, x):
        x = self.pos_conv_embed(x)
        x = self.layer_norm(x)
        return x

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        순방향 패스

        Args:
            x: 입력 텐서 (B, N, C)
            attention_mask: 어텐션 마스크 (B, N, N)

        Returns:
            출력 텐서 (B, N, C)
        """

        x = self.post_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
if __name__ == '__main__':
    # 하이퍼파라미터 정의
    batch_size = 2
    sequence_length = 50
    embedding_dim = 768  # 예시 값
    num_embeddings = 128  # 예시 값
    kernel_size = 3
    num_conv_pos_embedding_groups = 16

    # ConvolutionalPositionalEmbedding 레이어 초기화
    pos_emb = ConvolutionalPositionalEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        groups=num_conv_pos_embedding_groups
    )

    transformer = Transformer(
        pos_conv_embed=pos_emb, 
        num_layers=4,
        embed_dim=768,
        num_heads=12,
        ff_hidden_dim=1024
    )

    # 임의의 입력 텐서 생성
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    # 포지셔널 임베딩 적용
    output_tensor = transformer(input_tensor)

    # 결과 확인
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
