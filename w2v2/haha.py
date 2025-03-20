import torch
from torch import nn
from torch.nn import Module

from typing import (
    Optional,
    Tuple,
    List,
    Any
)

from einops import rearrange
from job import *
import numpy

from config import base, pretrainer_config
def _compute_mask_indices(
	shape: Tuple[int, int],
	mask_prob: float,
	mask_length: int,
	min_masks: int = 0,
	lengths: Optional[torch.Tensor] = None,
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

def _buffered_arange(max: int) -> torch.Tensor:
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
	input: torch.Tensor, num_negatives: int, cross_sample_negatives: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
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

class GumbelVectorQuantizer(Module):
    def __init__(self,
                dim: int = 512,
                codevector_dim: int = 256,
                groups: int = 2,
                num_vars: int = 320,
                temperature: float = 2.0,
                dropout: float = 0.0
        ) -> None:
        super().__init__()
        self.num_groups = groups
        self.num_vars = num_vars
        self.temperature = temperature

        if codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`codevector_dim` {codevector_dim} must be divisible "
                f"by `num_codevector_groups` {self.num_groups} for concatenation"
            )

        self.codevectors = nn.Parameter(
            torch.FloatTensor(
                1, self.num_groups * self.num_vars, codevector_dim // self.num_groups
            ).uniform_()
        )

        if 0 < dropout < 1:
            self.dropout = nn.Dropout(dropout)

        self.weight_proj = nn.Linear(dim, self.num_groups * self.num_vars)

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(
            -torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)
        ).sum()
        return perplexity
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape

        if hasattr(self, "dropout"):
            hidden_states = self.dropout(hidden_states)

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(
            batch_size * sequence_length * self.num_groups, -1
        )

        if self.training:
            # 학습 모드: Gumbel-Softmax로 hard one-hot 선택 (미분 가능 방식)
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)
            # Soft distribution: (BT, G, V)
            codevector_soft_dist = torch.softmax(
                hidden_states.view(
                    batch_size * sequence_length, self.num_groups, -1
                ).float(),
                dim=-1,
            )
            codevector_dist = codevector_soft_dist
        else:
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            # (BT, G, V)
            codevector_probs = codevector_probs.view(
                batch_size * sequence_length, self.num_groups, -1
            )
            codevector_dist = codevector_probs

        codevectors_per_group = (
            codevector_probs.view(batch_size * sequence_length, -1, 1)
            * self.codevectors
        )
        # (B, T, GV, d/G) --> (BT, G, V, d/G)
        codevectors = codevectors_per_group.view(
            batch_size * sequence_length, self.num_groups, self.num_vars, -1
        )
        # (BT, G, V, d/G) --> (BT, G, d/G) --> (B, T, d)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, codevector_dist

class Wav2Vec2PreTrainer(Module):
    def __init__(
        self,
        w2v2: Wav2Vec2Model,
        proj_codevector_dim: int = 256,  # f
        codevector_dim: int = 256,  # d
        num_codevector_groups: int = 2,  # G
        num_codevectors_per_group: int = 320,  # V
        gumbel_softmax_temperature: float = 2.0,
        quantizer_dropout: float = 0,
        mask_time_prob: float = 0.065,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
		**kwargs: Any,
    ) -> None:
        super().__init__()
        self.w2v2 = w2v2
        
        self.quantizer = GumbelVectorQuantizer(
			dim=self.w2v2.extractor_embed_dim,
			codevector_dim=codevector_dim,
			groups=num_codevector_groups,
			num_vars=num_codevectors_per_group,
			temperature=gumbel_softmax_temperature,
			dropout=quantizer_dropout,
		)
        self.proj_hid = nn.Linear(self.w2v2.encoder_embed_dim, proj_codevector_dim)
        self.proj_q = nn.Linear(codevector_dim, proj_codevector_dim)

        self.mask_time_prob = float(mask_time_prob)
        self.mask_time_length = int(mask_time_length)
        self.mask_time_min_masks = int(mask_time_min_masks)
        self.masked_spec_embed = nn.Parameter(
            torch.FloatTensor(self.w2v2.encoder_embed_dim).uniform_()
        )
        assert self.mask_time_prob > 0

    
    def forward(
        self, input: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, z_pre, lengths, attention_mask = self.w2v2.extract_latents(
            input, lengths
        )

        mask_time_indices = _compute_mask_indices(
            tuple(z.shape[:2]),
            self.mask_time_prob,
            self.mask_time_length,
            min_masks=self.mask_time_min_masks,
            lengths=lengths,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(device=z.device)
        z[mask_time_indices] = self.masked_spec_embed.to(z.dtype)

        c = self.w2v2.extract_contexts_from_latents(z, attention_mask)

        # Quantized representations
        q, q_dist = self.quantizer(z_pre)
        # print(q_dist)
        # print(q)
        q = q.to(self.proj_q.weight.dtype)

        # Projection
        c = self.proj_hid(c)
        q = self.proj_q(q)
        return c, q, q_dist, mask_time_indices
    

    
    @staticmethod
    def compute_contrastive_loss(
        c: torch.Tensor,
        q: torch.Tensor,
        mask_time_indices: torch.Tensor,
        num_negatives: int = 100,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[q, q_neg]` and `c`. Additionally, temperature can be applied.
        """
        # c, q: (B, T, D)

        # (K, B, T, D)
        q_neg, _ = _sample_negatives(q, num_negatives)

        # (K+1, B, T, D)
        q_all = torch.cat([q[None, :], q_neg], dim=0)

        # (K+1, B, T)
        logits = (
            torch.cosine_similarity(c.float(), q_all.float(), dim=-1).type_as(q_all)
            / temperature
        )

        # (K, B, T)
        neg_is_pos = (q == q_neg).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        # (K+1, B, T) --> (T, B, K+1) --> (TB, K+1)
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))

        # (B, T) --> (T, B) --> (TB,)
        # -100 corresponds to `ignore_index` in the arguments for `nn.functional.cross_entropy`
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

        contrastive_loss = nn.functional.cross_entropy(
            logits.float(), target, reduction="sum"
        ) / float(mask_time_indices.sum())

        return contrastive_loss

    @staticmethod
    def compute_diversity_loss(
        probs: torch.Tensor, mask: Optional[torch.Tensor], num_codevectors: int
    ) -> torch.Tensor:
        """
        probs: (BT, G, V)
        mask: (B, T)
        """

        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        # (G, V) --> (G,) --> ()
        perplexity = torch.exp(
            -torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)
        ).sum()
        diversity_loss = (num_codevectors - perplexity) / num_codevectors

        return diversity_loss

    def decay_gumbel_softmax_temperature(
        self, decay_rate: float, min_temperature: float
    ) -> None:
        self.quantizer.temperature = max(
            self.quantizer.temperature * decay_rate, min_temperature
        )
