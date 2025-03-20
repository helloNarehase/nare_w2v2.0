import math
import numpy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, functional as F
from typing import Any, List, Optional, Tuple

from Transformer import (
    Attention,
    FeedForward,
    Transformer,
    TransformerBlock,
    ConvolutionalPositionalEmbedding
)
from GumbelVectorQuantizer import GumbelVectorQuantizer


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
    lengths: Optional[Tensor] = None,
) -> numpy.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
            the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
            independently generated mask spans of length `mask_length` is computed by
            `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
            actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        lengths: Indicates the valid length of each sequence in the batch.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = numpy.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        lengths.detach().tolist()
        if lengths is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = numpy.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = numpy.random.choice(
            numpy.arange(input_length - (mask_length - 1)),
            num_masked_span,
            replace=False,
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = numpy.concatenate(
            [
                spec_aug_mask_idx,
                numpy.ones(max_num_masked_span - num_masked_span, dtype=numpy.int32)
                * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = numpy.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = numpy.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(
        batch_size, max_num_masked_span * mask_length
    )

    # add offset to the starting indexes so that indexes now create a span
    offsets = numpy.arange(mask_length)[None, None, :]
    offsets = numpy.broadcast_to(
        offsets, (batch_size, max_num_masked_span, mask_length)
    ).reshape(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = (
            sequence_length - 1
        )

    # scatter indices to mask
    numpy.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def _buffered_arange(max: int) -> Tensor:
    """Compute arange using a buffered tensor across function calls.
    Produces same result as torch.arange(end=max).

    Args:
        max (int): Ending value for arange.
    """
    if not hasattr(_buffered_arange, "buf"):
        _buffered_arange.buf = torch.LongTensor()
    if max > _buffered_arange.buf.numel():
        _buffered_arange.buf.resize_(max)
        torch.arange(max, out=_buffered_arange.buf)
    return _buffered_arange.buf[:max]


def _sample_negatives(
	input: Tensor, num_negatives: int, cross_sample_negatives: int = 0
) -> Tuple[Tensor, Tensor]:
	"""Sample negative examples from masked input.

	Args:
		input (Tensor): Tensor of dimension `(batch, frame, dim)`.
		num_negatives (int): Number of negative examples to sample.
		cross_sample_negatives (int): Number of negative examples to cross sample.

	Returns:
		(Tensor, Tensor):
		Tensor
			The negative samples.
		Tensor
			The indices of the negative samples.
	"""
	if num_negatives == 0 and cross_sample_negatives == 0:
		return (
			torch.zeros(0).to(input.device, input.dtype),
			torch.zeros(0).to(input.device, input.dtype),
		)

	B, T, D = input.shape
	input = input.view(-1, D)

	cross_high = T * B
	high = T

	assert high > 1

	if num_negatives > 0:
		tszs = _buffered_arange(T).unsqueeze(-1).expand(-1, num_negatives).flatten()

		neg_idxs = torch.randint(low=0, high=high - 1, size=(B, num_negatives * T))
		neg_idxs[neg_idxs >= tszs] += 1

	if cross_sample_negatives > 0:
		tszs = (
			_buffered_arange(T)
			.unsqueeze(-1)
			.expand(-1, cross_sample_negatives)
			.flatten()
		)

		cross_neg_idxs = torch.randint(
			low=0, high=cross_high - 1, size=(B, cross_sample_negatives * T)
		)
		cross_neg_idxs[cross_neg_idxs >= tszs] += 1

	if num_negatives > 0:
		neg_idxs = neg_idxs + (torch.arange(B).unsqueeze(1) * high)
	else:
		neg_idxs = cross_neg_idxs

	if cross_sample_negatives > 0 and num_negatives > 0:
		neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

	negs = input[neg_idxs.view(-1)]
	negs = negs.view(B, T, num_negatives + cross_sample_negatives, D).permute(
		2, 0, 1, 3
	)  # NxBxCxT

	return negs, neg_idxs


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class Wav2Vec2Model(nn.Module):
    """
    Wav2Vec2.0 model implementation
    
    Args:
        feature_extractor_conv_layers (List[Tuple[int, int, int]]): List of tuples defining the conv layers
        feature_extractor_dropout (float): Dropout probability for feature extractor
        feature_extractor_mode (str): Mode for feature extractor ('default' or 'layer_norm')
        feature_extractor_conv_bias (bool): Whether to use bias in conv layers
        encoder_embed_dim (int): Embedding dimension for transformer encoder
        encoder_num_layers (int): Number of transformer layers
        encoder_num_heads (int): Number of attention heads
        encoder_ff_hidden_dim (int): Hidden dimension for feed-forward network
        encoder_dropout (float): Dropout probability for transformer encoder
        encoder_ff_dropout (float): Dropout probability for feed-forward network
        encoder_layer_norm_eps (float): Epsilon for layer normalization
        quantizer_dim (int): Input dimension for quantizer
        quantizer_codevector_dim (int): Dimension of each codevector
        quantizer_groups (int): Number of groups in the quantizer
        quantizer_num_vars (int): Number of variables per group
        quantizer_temp (float): Temperature for Gumbel-Softmax
        quantizer_dropout (float): Dropout probability for quantizer
        pos_conv_kernel (int): Kernel size for positional embedding
        pos_conv_groups (int): Groups for positional embedding
        mask_prob (float): Probability of applying masking during training
        mask_length (int): Length of the mask spans
        mask_min_masks (int): Minimum number of mask spans
        mask_selection (str): Method for selecting masks ('static', 'uniform', 'normal', 'poisson')
        mask_other (float): Probability of replacing a mask with another mask
        mask_channel_prob (float): Probability of masking channels
        mask_channel_length (int): Length of channel mask spans
        mask_channel_min_masks (int): Minimum number of channel mask spans
        mask_channel_selection (str): Method for selecting channel masks ('static', 'uniform', 'normal', 'poisson')
        mask_channel_other (float): Probability of replacing a channel mask with another mask
        num_negatives (int): Number of negative samples
        cross_sample_negatives (int): Number of negative samples to cross sample
    """
    def __init__(
        self,
        feature_extractor_conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
        feature_extractor_dropout: float = 0.0,
        feature_extractor_mode: str = "default",
        feature_extractor_conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 12,
        encoder_ff_hidden_dim: int = 3072,
        encoder_dropout: float = 0.1,
        encoder_ff_dropout: float = 0.1,
        encoder_layer_norm_eps: float = 1e-5,
        quantizer_dim: int = 768,
        quantizer_codevector_dim: int = 256,
        quantizer_groups: int = 2,
        quantizer_num_vars: int = 320,
        quantizer_temp: float = 2.0,
        quantizer_dropout: float = 0.1,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_min_masks: int = 2,
        num_negatives: int = 100,
        cross_sample_negatives: int = 0,
    ):
        super().__init__()
        
        # Feature Extractor
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_extractor_conv_layers,
            dropout=feature_extractor_dropout,
            mode=feature_extractor_mode,
            conv_bias=feature_extractor_conv_bias,
        )
        
        # Transformer input projection
        self.layer_norm = nn.LayerNorm(feature_extractor_conv_layers[-1][0])
        self.post_extract_proj = nn.Linear(feature_extractor_conv_layers[-1][0], encoder_embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(encoder_dropout)
        
        # Positional Embedding
        self.pos_conv = ConvolutionalPositionalEmbedding(
            num_embeddings=pos_conv_kernel,
            embedding_dim=encoder_embed_dim,
            kernel_size=pos_conv_kernel,
            groups=pos_conv_groups,
        )
        
        # Encoder
        self.encoder = Transformer(
            pos_conv_embed=self.pos_conv,
            num_layers=encoder_num_layers,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            ff_hidden_dim=encoder_ff_hidden_dim,
            dropout=encoder_dropout,
            ff_dropout=encoder_ff_dropout,
            layer_norm_eps=encoder_layer_norm_eps,
        )
        
        # Quantizer
        self.quantizer = GumbelVectorQuantizer(
            dim=quantizer_dim,
            codevector_dim=quantizer_codevector_dim,
            groups=quantizer_groups,
            num_vars=quantizer_num_vars,
            temperature=quantizer_temp,
            dropout=quantizer_dropout,
        )
        
        # Project quantized representations to encoder dimension
        self.project_q = nn.Linear(quantizer_codevector_dim, encoder_embed_dim)
        
        # Final projection
        self.final_proj = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        
        # Masking parameters
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_min_masks = mask_min_masks
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives
        
    def forward_features(self, source: Tensor) -> Tensor:
        """
        Forward pass through the feature extractor
        
        Args:
            source (Tensor): Input audio waveform (B, T)
            
        Returns:
            Tensor: Extracted features (B, T', C)
        """
        features = self.feature_extractor(source)  # B, C, T'
        features = features.transpose(1, 2)  # B, T', C
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)
        return features
    
    def apply_mask(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply masking to the input features
        
        Args:
            x (Tensor): Input features (B, T, C)
            padding_mask (Optional[Tensor]): Padding mask (B, T)
            
        Returns:
            Tuple[Tensor, Tensor]: Masked features and mask
        """
        B, T, C = x.shape
        
        if self.mask_prob > 0:
            mask_indices = torch.from_numpy(
                _compute_mask_indices(
                    shape=(B, T),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    min_masks=self.mask_min_masks,
                    lengths=None if padding_mask is None else (~padding_mask).sum(1),
                )
            ).to(x.device)
            
            # Apply mask
            mask = mask_indices.bool()
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
            return x, mask
        else:
            # No masking
            return x, torch.zeros(B, T).bool().to(x.device)
    
    def forward_transformer(self, features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the transformer encoder
        
        Args:
            features (Tensor): Input features (B, T', C)
            mask (Optional[Tensor]): Attention mask (B, T')
            
        Returns:
            Tensor: Encoded features (B, T', C)
        """
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # B, 1, 1, T'
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
        else:
            attention_mask = None
            
        features = self.dropout(features)
        x = self.encoder(features, attention_mask)
        return x
    
    def quantize(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Quantize the encoder output
        
        Args:
            x (Tensor): Encoder output (B, T', C)
            mask (Optional[Tensor]): Mask (B, T')
            
        Returns:
            Tuple[Tensor, Tensor]: Quantized features and perplexity
        """
        quantized, perplexity = self.quantizer(x, mask)
        quantized = self.project_q(quantized)
        return quantized, perplexity
    
    def sample_negatives(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample negative examples for contrastive learning
        
        Args:
            x (Tensor): Encoder output (B, T', C)
            
        Returns:
            Tuple[Tensor, Tensor]: Negative samples and indices
        """
        return _sample_negatives(
            x, 
            self.num_negatives, 
        )
    
    def forward(
        self,
        source: Tensor,
        padding_mask: Optional[Tensor] = None,
        mask: bool = True,
        return_quantized: bool = False,
        return_negatives: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass for the Wav2Vec2 model
        
        Args:
            source (Tensor): Input audio waveform (B, T)
            padding_mask (Optional[Tensor]): Padding mask (B, T)
            mask (bool): Whether to apply masking
            return_quantized (bool): Whether to return quantized features
            return_negatives (bool): Whether to return negative samples
            
        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
                - Encoded features
                - Quantized features (if return_quantized=True)
                - Perplexity (if return_quantized=True)
                - Negative samples (if return_negatives=True)
                - Mask (if mask=True)
        """
        # Extract features
        features = self.forward_features(source)
        
        # Apply masking if needed
        mask_output = None
        if mask:
            features, mask_output = self.apply_mask(features, padding_mask)
        
        # Forward through transformer
        x = self.forward_transformer(features, padding_mask)
        
        # Quantize
        quantized, perplexity = self.quantize(x, padding_mask)
        
        # Sample negatives if needed
        negatives = None
        neg_idxs = None
        if return_negatives:
            negatives, neg_idxs = self.sample_negatives(x)
        
        # Final projection
        x = self.final_proj(x)
        
        outputs = (x,)
        
        if return_quantized:
            outputs += (quantized, perplexity)
        else:
            outputs += (None, None)
            
        if return_negatives:
            outputs += (negatives,)
        else:
            outputs += (None,)
            
        if mask:
            outputs += (mask_output,)
        else:
            outputs += (None,)
            
        return outputs

class ContrastiveLossModule(nn.Module):
    def __init__(self, num_negatives: int = 100, temperature: float = 0.1):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature

    @staticmethod
    def compute_contrastive_loss(
        c: Tensor,
        q: Tensor,
        mask_time_indices: Tensor,
        num_negatives: int = 100,
        temperature: float = 0.1,
    ) -> Tensor:
        """
        Compute contrastive loss using cosine similarity.
        
        Args:
            c (Tensor): Context features, shape (B, T, D)
            q (Tensor): Positive quantized features, shape (B, T, D)
            mask_time_indices (Tensor): Mask for valid time steps, shape (B, T)
            num_negatives (int): Number of negative samples to draw.
            temperature (float): Scaling factor for logits.
            
        Returns:
            Tensor: Contrastive loss scalar.
        """
        # Sample negative examples: (K, B, T, D)
        q_neg, _ = _sample_negatives(q, num_negatives)
        
        # Concatenate positive and negative examples:
        # q_all: (K+1, B, T, D) with 첫번째가 positive, 나머지가 negatives
        q_all = torch.cat([q[None, :], q_neg], dim=0)
        
        # Compute cosine similarity between context c and each example in q_all.
        # c.unsqueeze(0): (1, B, T, D) broadcasted to (K+1, B, T, D)
        # 결과 logits: (K+1, B, T)
        logits = torch.cosine_similarity(c.unsqueeze(0).float(), q_all.float(), dim=-1)
        logits = logits / temperature

        # 만약 negative sample이 positive와 동일하다면, 해당 유사도를 -inf로 처리합니다.
        # 비교를 위해 q와 q_neg의 모양을 맞춤.
        # q: (1, B, T, D), q_neg: (K, B, T, D) -> 결과 neg_is_pos: (K, B, T)
        neg_is_pos = (q.unsqueeze(0) == q_neg).all(dim=-1)
        if neg_is_pos.any():
            # logits[1:]는 negatives에 해당하는 logits입니다.
            logits[1:][neg_is_pos] = float("-inf")
        
        # logits: (K+1, B, T) --> (T, B, K+1) --> (B*T, K+1)
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        
        # target: 유효한 시점에서 positive (index 0)를 정답으로, 무효한 시점은 -100 (ignore_index)
        # mask_time_indices: (B, T) --> (T, B) --> (B*T,)
        # mask_time_indices가 1인 경우 정답은 0, 0인 경우 ignore (-100)
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        
        # Cross entropy loss 계산 (ignore_index=-100에 해당하는 값은 무시)
        loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum") / float(mask_time_indices.sum())
        return loss

    @staticmethod
    def compute_diversity_loss(
        probs: Tensor, mask: Optional[Tensor], num_codevectors: int
    ) -> Tensor:
        """
        Compute diversity loss to encourage usage of all code vectors.
        
        Args:
            probs (Tensor): Probabilities, shape (B*T, G, V) 또는 (BT, G, V)
            mask (Optional[Tensor]): Mask, shape (B, T)
            num_codevectors (int): 총 codevector의 수.
            
        Returns:
            Tensor: Diversity loss scalar.
        """
        # probs: (BT, G, V)
        if mask is not None:
            # mask: (B, T) -> (B*T, 1, 1)로 확장
            mask_extended = mask.flatten()[:, None, None].expand_as(probs)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)
        
        # 각 그룹 G마다의 perplexity 계산: exp(-sum(p * log(p)))
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        diversity_loss = (num_codevectors - perplexity) / num_codevectors
        
        return diversity_loss

    def decay_gumbel_softmax_temperature(self, decay_rate: float, min_temperature: float) -> None:
        """
        Decay the temperature for the Gumbel softmax in the quantizer.
        """
        # self.quantizer.temperature을 업데이트한다고 가정합니다.
        # (quantizer가 모듈 내 속성으로 구현되어 있어야 합니다.)
        self.temperature = max(self.temperature * decay_rate, min_temperature)


# Loss function for Wav2Vec2 with contrastive learning
class Wav2Vec2ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Wav2Vec2 model
    
    Args:
        temperature (float): Temperature for contrastive loss
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(
        self,
        x: Tensor,
        quantized: Tensor,
        negatives: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Wav2Vec2 contrastive loss
        
        Args:
            x (Tensor): Encoded features (B, T, C)
            quantized (Tensor): Quantized features (B, T, C)
            negatives (Tensor): Negative samples (N, B, T, C)
            mask (Optional[Tensor]): Mask (B, T)
            
        Returns:
            Tuple[Tensor, Tensor]: Loss and accuracy
        """
        # Normalize features
        x = F.normalize(x, dim=-1)
        quantized = F.normalize(quantized, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # Get dimensions
        batch_size, time_length, dim = x.shape
        neg_samples = negatives.shape[0]
        
        # Compute similarity for positives
        pos_similarity = torch.bmm(
            x.reshape(-1, dim).unsqueeze(1),
            quantized.reshape(-1, dim).unsqueeze(2)
        ).squeeze(-1) / self.temperature
        
        # Compute similarity for negatives
        neg_similarity = torch.bmm(
            x.reshape(-1, dim).unsqueeze(1),
            negatives.permute(1, 2, 0, 3).reshape(-1, neg_samples, dim).transpose(1, 2)
        ) / self.temperature
        
        # Combine positive and negative similarities
        print(pos_similarity.shape, neg_similarity.shape)
        logits = torch.cat([pos_similarity, neg_similarity], dim=1)
        
        # Targets (0 is the positive example)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.reshape(-1)
            logits = logits[mask]
            targets = targets[mask]
        
        # Compute loss
        loss = self.cross_entropy(logits, targets)
        
        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).sum()
            total = targets.numel()
            accuracy = correct.float() / total if total > 0 else torch.tensor(0.0, device=correct.device)
        
        return loss, accuracy



if __name__ == "__main__":
    # Instantiate the model
    model = Wav2Vec2Model()
    
    # Create a sample input
    batch_size = 2
    seq_length = 10000  # Raw audio samples
    audio_input = torch.randn(batch_size, seq_length)
    
    # Forward pass with masking and negatives
    encoded_features, quantized_features, perplexity, negatives, mask = model(
        audio_input, 
        mask=True,               # 마스킹 사용
        return_quantized=True,   # 양수(positive) 샘플 반환
        return_negatives=True    # 음수(negative) 샘플 반환
    )

    # Compute contrastive loss
    contrastive_loss = ContrastiveLossModule.compute_contrastive_loss(
        encoded_features, quantized_features, mask, num_negatives=100, temperature=0.1
    )

    print("Contrastive Loss:", contrastive_loss.item())