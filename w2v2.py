import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, functional as F

from Transformer import (
    Attention,
    FeedForward,
    Transformer,
    TransformerBlock,
    ConvolutionalPositionalEmbedding
)
from GumbelVectorQuantizer import GumbelVectorQuantizer


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

        self.aux = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        
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
            attention_mask = (1.0 - attention_mask) * -10000.0
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
    
    def forward(
        self,
        source: Tensor,
        mask: Optional[Tensor] = None,
        return_quantized: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass for the Wav2Vec2 model
        
        Args:
            source (Tensor): Input audio waveform (B, T)
            mask (Optional[Tensor]): Attention mask (B, T')
            return_quantized (bool): Whether to return quantized features
            
        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
                - Encoded features
                - Quantized features (if return_quantized=True)
                - Perplexity (if return_quantized=True)
        """
        # Extract features
        features = self.forward_features(source)
        
        # Forward through transformer
        x = self.forward_transformer(features, mask)
        
        # Quantize
        quantized, perplexity = self.quantize(x, mask)
        
        # Final projection
        x = self.aux(x)
        
        if return_quantized:
            return x, quantized, perplexity
        else:
            return x, None, None


# Loss function for Wav2Vec2
class Wav2Vec2Loss(nn.Module):
    """
    Contrastive loss for Wav2Vec2 model
    
    Args:
        temperature (float): Temperature for contrastive loss
        negative_samples (int): Number of negative samples
    """
    def __init__(self, temperature: float = 0.1, negative_samples: int = 100):
        super().__init__()
        self.temperature = temperature
        self.negative_samples = negative_samples
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(
        self,
        x: Tensor,
        quantized: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Wav2Vec2 loss
        
        Args:
            x (Tensor): Encoded features (B, T, C)
            quantized (Tensor): Quantized features (B, T, C)
            mask (Optional[Tensor]): Mask (B, T)
            
        Returns:
            Tuple[Tensor, Tensor]: Loss and accuracy
        """
        batch_size, time_steps, dimension = x.shape
        
        # Flatten time dimension
        x = x.reshape(-1, dimension)  # (B*T, C)
        quantized = quantized.reshape(-1, dimension)  # (B*T, C)
        
        # Normalize features
        x = F.normalize(x, dim=-1)
        quantized = F.normalize(quantized, dim=-1)
        
        # Compute similarity
        similarity = torch.matmul(x, quantized.transpose(0, 1)) / self.temperature  # (B*T, B*T)
        
        # Set up positive and negative samples
        targets = torch.arange(similarity.shape[0], device=similarity.device)  # (B*T,)
        
        # Compute loss
        loss = self.cross_entropy(similarity, targets)
        
        # Compute accuracy
        with torch.no_grad():
            predictions = similarity.argmax(dim=-1)
            correct = (predictions == targets).sum()
            total = targets.numel()
            accuracy = correct.float() / total
        
        return loss, accuracy


# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = Wav2Vec2Model()
    
    # Create a sample input
    batch_size = 2
    seq_length = 10000  # Raw audio samples
    audio_input = torch.randn(batch_size, seq_length)
    
    # Forward pass
    encoded_features, quantized_features, perplexity = model(audio_input, return_quantized=True)
    
    # Print shapes
    print("Input shape:", audio_input.shape)  # (B, T)
    print("Encoded features shape:", encoded_features.shape)  # (B, T', C)
    print("Quantized features shape:", quantized_features.shape)  # (B, T', C)
    print("Perplexity shape:", perplexity.shape)  # (G, V)
    
    # Compute loss
    loss_fn = Wav2Vec2Loss()
    loss, accuracy = loss_fn(encoded_features, quantized_features)
    
    print("Loss:", loss.item())
    print("Accuracy:", accuracy.item())