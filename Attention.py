import torch
from torch import nn, Tensor
from typing import Optional
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        hidden_dim_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.hidden_dim_dense = nn.Linear(embed_dim, hidden_dim)
        self.hidden_dim_dropout = nn.Dropout(hidden_dim_dropout)
        self.output_dense = nn.Linear(hidden_dim, embed_dim)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        x = self.hidden_dim_dense(x)
        x = self.hidden_dim_dropout(x)
        x = F.relu(x)
        x = self.output_dense(x)
        return self.output_dropout(x)
    
class Attention(nn.Module):
    """
    Self-Attention 모델
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        local_window_size: Optional[int] = None,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "err embed_dim, num_heads"

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.local_window_size = local_window_size

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        순방향 패스

        Args:
            x: 입력 텐서 (B, N, C)
            attention_mask: 어텐션 마스크 (B, N, N)
            local_attention_mask: 로컬 어텐션 마스크 (window_size, window_size)

        Returns:
            출력 텐서 (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv_linear(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_linear(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        ff_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attention = Attention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim, ff_dropout, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.drop_one = nn.Dropout(dropout)
        self.drop_two = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        순방향 패스

        Args:
            x: 입력 텐서 (B, N, C)
            attention_mask: 어텐션 마스크 (B, N, N)

        Returns:
            출력 텐서 (B, N, C)
        """
        x = self.layer_norm1(x)
        x = self.drop_one(self.attention(x, attention_mask)) + x
        x = self.layer_norm2(x)
        x = self.drop_two(self.feed_forward(x)) + x
        return x
    
if __name__ == "__main__":
    # 하이퍼파라미터
    embed_dim = 256
    num_heads = 8
    ff_hidden_dim = 1024
    dropout = 0.1
    batch_size = 3
    seq_length = 16

    # 입력 텐서 생성
    x = torch.randn(batch_size, seq_length, embed_dim)

    # 어텐션 마스크 생성 (예: 패딩 마스크)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    attention_mask = torch.triu(attention_mask, 1)
    for b in range(batch_size):
        # 예시로, 시퀀스 길이의 절반 이후를 패딩으로 마스킹
        attention_mask[b, :, seq_length // 2 :] = 0
        attention_mask[b, seq_length // 2 :, :] = 0

    # TransformerBlock 모델 초기화
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)

    # 순방향 패스
    output = transformer_block(x, attention_mask)

    # 결과 출력
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
