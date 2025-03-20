import torch
from torch import nn, Tensor
from einops import rearrange
from typing import Optional, Tuple


class GumbelVectorQuantizer(nn.Module):
    """Gumbel-Softmax 기반 벡터 양자화 모듈.

    Args:
        input_dim (int): 입력 특징 차원.
        codevector_dim (int): 최종 코드벡터 차원.
        num_codevector_groups (int): 코드벡터 그룹 수 (G).
        num_codevectors_per_group (int): 그룹당 코드벡터 개수 (V).
        temperature (float): Gumbel-Softmax 온도.
        dropout (float): 드롭아웃 확률 (0이면 미적용).
    """

    def __init__(self,
                 dim: int = 512,
                 codevector_dim: int = 256,
                 groups: int = 2,
                 num_vars: int = 320,
                 temperature: float = 2.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.num_groups = groups
        self.num_vars = num_vars
        self.temperature = temperature

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`codevector_dim` {codevector_dim} must be divisible "
                f"by `num_codevector_groups` {self.num_groups} for concatenation"
            )

        self.code_dim = codevector_dim // self.num_groups

        # 입력을 각 그룹의 코드벡터 후보 수로 변환
        self.weight_proj = nn.Linear(dim, self.num_groups * self.num_vars)

        # 드롭아웃 (0 < dropout < 1 인 경우)
        if 0 < dropout < 1:
            self.dropout = nn.Dropout(dropout)

        # 코드북: 전체 코드벡터 개수 x 각 코드벡터 차원 (nn.Embedding 사용)
        self.codebook = nn.Embedding(self.num_groups * self.num_vars, self.code_dim)
        nn.init.uniform_(self.codebook.weight)

    @staticmethod
    def _compute_perplexity(probs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """엔트로피 기반으로 퍼플렉서티 계산.

        Args:
            probs (Tensor): 분포 (BT, G, V)
            mask (Optional[Tensor]): (B, T) 형태의 유효 시퀀스 위치 마스크
        
        Returns:
            Tensor: perplexity (G, V)
        """
        if mask is not None:
            # mask: (B, T) -> (B*T,)
            mask = mask.flatten()
            valid_probs = probs[mask]
            num_values = valid_probs.size(0)
            perplexity = valid_probs.sum(dim=0) / num_values  # (G, V)
        else:
            perplexity = probs.mean(dim=0)
        return perplexity

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """전방 전달.

        Args:
            hidden_states (Tensor): 입력 특징 (B, T, input_dim).
            mask (Optional[Tensor]): (B, T) 형태의 유효 시퀀스 위치 마스크.
        
        Returns:
            Tuple[Tensor, Tensor]:
                - 양자화된 코드벡터 (B, T, codevector_dim)
                - 퍼플렉서티 (G, V)
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # 드롭아웃 적용 (있다면)
        if hasattr(self, "dropout"):
            hidden_states = self.dropout(hidden_states)

        # 선형 변환 후 (B, T, G*V) -> (B*T, G, V)
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = rearrange(hidden_states, 'b t (g v) -> (b t) g v', g=self.num_groups)

        if self.training:
            # 학습 모드: Gumbel-Softmax로 hard one-hot 선택 (미분 가능 방식)
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            )
            # soft 분포 (엔트로피 계산용)
            codevector_soft_dist = torch.softmax(hidden_states.float(), dim=-1)
            codevector_dist = codevector_soft_dist
        else:
            # 평가 모드: argmax 후 one-hot 생성 (미분 불가능)
            codevector_idx = hidden_states.argmax(dim=-1)  # (B*T, G)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape)
            codevector_probs.scatter_(-1, codevector_idx.unsqueeze(-1), 1.0)
            codevector_dist = codevector_probs

        # 퍼플렉서티 계산 (mask가 있다면 전달)
        perplexity = self._compute_perplexity(codevector_dist, mask)

        # 코드북 재배열: (num_groups * num_vars, code_dim) -> (G, V, code_dim)
        codebook = rearrange(self.codebook.weight, '(g v) d -> g v d', g=self.num_groups)

        # 각 그룹별로 코드북과 분포의 가중합 계산
        # codevector_probs: (B*T, G, V), codebook: (G, V, code_dim)
        group_vectors = torch.einsum('bgv,gvd->bgd', codevector_probs, codebook)
        # 재배열: (B*T, G, code_dim) -> (B, T, G*code_dim)
        code_vectors = rearrange(group_vectors, '(b t) g d -> b t (g d)',
                                  b=batch_size, t=sequence_length)

        return code_vectors, perplexity


# 테스트 코드
if __name__ == "__main__":
    # 모델 인스턴스 생성 (개별 파라미터 전달)
    model = GumbelVectorQuantizer(
        dim=512,
        codevector_dim=256,
        groups=2,
        num_vars=320,
        temperature=2.0,
        dropout=0.1
    )

    batch_size, seq_len, feature_dim = 2, 5, 512
    hidden_states = torch.randn(batch_size, seq_len, feature_dim)
    lengths = torch.tensor([5, 3])  # 각 배치별 실제 시퀀스 길이

    # mask 생성: (B, T), 유효한 시퀀스 위치는 True로 설정
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    code_vectors, perplexity = model(hidden_states, mask)
    print("Code Vectors Shape:", code_vectors.shape)  # 예상: (2, 5, 256)
    print("Perplexity Shape:", perplexity.shape)        # 예상: (2, 320)
