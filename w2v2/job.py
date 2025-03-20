import torch
from torch.nn import Module
from torch import nn

from typing import (
    Optional,
    Tuple,
    List
)

import einops

class LN(nn.LayerNorm):
    def forward(self, x):
        x = x.transpose(-2, -1) # B C L -> B L C
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.transpose(-2, -1) # B L C -> B C L
        return x
    
class ConvBlock(Module):
    def __init__(self, 
                inp_c, 
                outp_c,
                ks,
                st,
                bias: bool,
                layer_norm: Optional[Module],
        ):
        super().__init__()
        self.conv = nn.Conv1d(
            inp_c, 
            outp_c,
            kernel_size=ks,
            stride=st,
            bias=bias
        )

        self.kernel = ks
        self.stride = st
        self.ln = layer_norm

    def forward(self, x, length:Optional[int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.conv(x)
        if self.ln is not None:
            x = self.ln(x)
        x = nn.functional.gelu(x)
        if length is not None:
            length = (
                torch.div(length - self.kernel, self.stride, rounding_mode="floor")
                + 1
            )
            length = torch.max(torch.zeros_like(length), length)
        return x, length
    
class FE(Module):
    def __init__(self, convs:nn.ModuleList):
        super().__init__()

        self.convs = convs
    
    def forward(self, x:torch.Tensor, length:Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim != 2:
            raise
        x = x.unsqueeze(1)
        for l in self.convs:
            x, length = l(x, length)
        x = x.transpose(1, 2)
        return x, length

class FP(Module):
    def __init__(
            self,
            inp_f,
            outp_f,
            dropout_p
        ):
        super().__init__()
        self.ln = nn.LayerNorm(inp_f)
        self.proj = nn.Linear(
            inp_f, outp_f
        )
        self.drop_p = nn.Dropout(
            dropout_p
        )

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pre = self.ln(x)
        x = self.proj(x_pre)
        x = self.drop_p(x)
        return x, x_pre

class ConvolutionalPositionalEmbedding(Module):
    def __init__(self, 
                embedding_dim: int, 
                kernel_size: int = 3, 
                groups: int = 1
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.parametrizations.weight_norm(
            self.conv, name="weight", dim=2
        )
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = nn.functional.gelu(x)
        x = x.transpose(-1, -2)
        return x    

class FeedForward(Module):
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
        x = nn.functional.gelu(x)
        x = self.output_dense(x)
        return self.output_dropout(x)
    
class Attention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        local_window_size: Optional[int] = None,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "err embed_dim, num_heads"
        
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.local_window_size = local_window_size

    def forward(self, x: torch.Tensor, att_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv_linear(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if att_mask is not None:
            attn = attn.masked_fill(att_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_linear(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(Module):
    def __init__(
        self,
        attention: Attention,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: FeedForward,
    ):
        super().__init__()
        self.att = attention
        self.ff = feed_forward
        self.ln_first = layer_norm_first

        self.dropout = nn.Dropout(dropout)
        
        self.ln = nn.LayerNorm(attention.embed_dim)
        self.final_ln = nn.LayerNorm(attention.embed_dim)

    def forward(self, x, att_mask):
        
        res = x

        if self.ln_first:
            x = self.ln(x)
        x = self.att(
            x,
            att_mask
        )
        x = self.dropout(x)
        x = res + x
        if self.ln_first:
            x = x + self.ff(self.final_ln(x))
        else:
            x = self.ln(x)
            x = self.final_ln(self.ff(x))

        return x

class Transformer(Module):
    def __init__(
        self,     
        pos_conv_embed: ConvolutionalPositionalEmbedding,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float
    ):
        super().__init__()    
        self.layer_norm_first = layer_norm_first
        self.pos_conv_embed = pos_conv_embed
        self.layer_drop = layer_drop
        self.layers = layers
        
        self.ln = nn.LayerNorm(pos_conv_embed.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _preprocess(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos_conv_embed(x)
        if self.layer_norm_first:
            x = self.ln(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x:torch.Tensor, att_mask:torch.Tensor):
        x = self._preprocess(x)
        for l in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = l(x, att_mask)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x
    
    def get_intermediate_outputs(
            self, 
            x:torch.Tensor,
            att_mask:torch.Tensor,
            num_layers: Optional[int] = None
        ):
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(
                    f"`num_layers` must be between [1, {len(self.layers)}]"
                )

        ret: List[torch.Tensor] = []
        x = self._preprocess(x)
        for layer in self.layers:
            x = layer(x, att_mask)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret
        
class Encoder(Module):
    def __init__(
            self,
            feature_projection: FP,
            transformer: Transformer,
            ):
        super().__init__()
        self.fp = feature_projection
        self.transformer = transformer
        
    def _preprocess(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None,) -> torch.Tensor:
        x, x_pre = self.fp(features)
        mask: Optional[torch.Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = (
                torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
                >= lengths[:, None]
            )
            x[mask] = 0.0
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, x_pre, mask
    
    def forward(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, _, mask = self._preprocess(features, lengths)
        x = self.transformer(x, mask)
        return x
    
    def extract_features(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[torch.Tensor]:
        x, _, masks = self._preprocess(features, lengths)
        return self.transformer.get_intermediate_outputs(
            x, att_mask=masks, num_layers=num_layers
        )
    
class EncoderLayer(Module):
    def __init__(
            self,
            att: Attention,
            dropout: float,
            layer_norm_first: bool,
            feed_forward: Module,
            ):
        super().__init__()
        self.att = att
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(att.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(att.embed_dim)

        
    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.att(
            x,
            att_mask=att_mask,
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x
        
class Wav2Vec2Model(Module):
    def __init__(
        self,
        feature_extractor: FE,
        encoder: Encoder,
        aux: Optional[FP] = None,
    ):
        super().__init__()
        self.fe = feature_extractor
        self.encoder = encoder
        self.aux = aux

    def extract_features(self, waveforms, lengths, num_layers):
        x, lengths = self.fe(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths
    
    def forward(
        self,
        source: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ):
        x, lengths = self.fe(source, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths
    
    def extract_latents(
        self, input: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z, lengths = self.fe(input, lengths)
        z, z_pre, att_mask = self.encoder._preprocess(z, lengths)
        return z, z_pre, lengths, att_mask
    
    def extract_contexts_from_latents(
        self, z: torch.Tensor, att_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        c = self.encoder.transformer(z, att_mask)
        if self.aux is not None:
            c = self.aux(c)
        return c



def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
) -> Encoder:
    feature_projection = FP(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(
        embed_dim, pos_conv_kernel, pos_conv_groups
    )

    encoder_layers = nn.ModuleList()
    for _ in range(num_layers):
        attention = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ff_interm_features,
            hidden_dim_dropout=ff_interm_dropout,
            output_dropout=dropout,
        )
        encoder_layers.append(
            EncoderLayer(
                att=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)



def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
) -> FE:
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LN(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvBlock(
                inp_c=in_channels,
                outp_c=out_channels,
                ks=kernel_size,
                st=stride,
                bias=bias,
                layer_norm=normalization,
            )
        )
        in_channels = out_channels
    return FE(nn.ModuleList(blocks))

def wav2vec2_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
) -> Wav2Vec2Model:

    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = (
            [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        )

    feature_extractor = _get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )
    encoder = _get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)

    model = Wav2Vec2Model(feature_extractor, encoder, aux)
    model.extractor_embed_dim = extractor_conv_layer_config[-1][0]
    model.encoder_embed_dim = encoder_embed_dim
    return model


if __name__ == "__main__":
    from config import base
    m = wav2vec2_model(**base["base"])
    x = torch.rand(2, 16000 * 15)
    h, l = m(x)
    # (B, T, D)
    print(h, h.shape)
