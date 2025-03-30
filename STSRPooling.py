import torch
import torch.nn as nn
from typing import Dict

class STSRPooling(nn.Module):
    """Spatio-Temporal Saliency Rescaled (STSR) Pooling with Attention."""

    def __init__(
        self,
        d_in: int,             # Input channel size
        n_clz: int,            # Number of classes
        d_attn: int = 8,       # Attention hidden size
        reduction_ratio: int = 4,  # Reduction ratio for channel attention
    ):
        super().__init__()
        self.d_in = d_in
        self.n_clz = n_clz

        # Time Attention
        self.time_attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.Linear(d_in, d_in // reduction_ratio),
            nn.ReLU(),
            nn.Linear(d_in // reduction_ratio, d_in),
            nn.Sigmoid(),
        )

        # Instance Classifier
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for STSR Pooling with time and channel attention.

        :param instance_embeddings: Tensor of shape [batch, n_channels, n_timesteps]
        :return: Dictionary containing bag logits, instance logits, attention maps, and TSR weights.
        """
        batch_size, n_channels, n_timesteps = instance_embeddings.size()

        # TSR weights
        time_relevance = instance_embeddings.mean(dim=1, keepdim=True)       # [B, 1, T]
        feature_relevance = instance_embeddings.mean(dim=2, keepdim=True)    # [B, C, 1]
        tsr_weight = time_relevance * feature_relevance                      # [B, C, T]

        # Attention Weights
        time_attention = self.time_attention_head(instance_embeddings.permute(0, 2, 1))  # [B, T, 1]
        time_attention = time_attention.permute(0, 2, 1)  # [B, 1, T]

        channel_attention = self.channel_attention(instance_embeddings.mean(dim=2))  # [B, C]
        channel_attention = channel_attention.unsqueeze(-1).expand_as(tsr_weight)   # [B, C, T]

        # Combine TSR and attentions
        final_weights = tsr_weight * time_attention.expand_as(tsr_weight) * channel_attention  # [B, C, T]
        weighted_instance_embeddings = instance_embeddings * final_weights  # [B, C, T]

        # Instance-level classification
        instance_logits = self.instance_classifier(weighted_instance_embeddings.permute(0, 2, 1))  # [B, T, n_clz]

        # Weighted average using time attention
        time_attention_expanded = time_attention.squeeze(1).unsqueeze(-1).expand_as(instance_logits)  # [B, T, n_clz]
        weighted_instance_logits = instance_logits * time_attention_expanded

        # Bag-level prediction
        bag_logits = weighted_instance_logits.mean(dim=1)  # [B, n_clz]

        return {
            "bag_logits": bag_logits,
            "instance_logits": instance_logits.permute(0, 2, 1),  # [B, n_clz, T]
            "time_attention": time_attention.squeeze(1),          # [B, T]
            "channel_attention": channel_attention.squeeze(2),    # [B, C]
            "tsr_weight": tsr_weight,                             # [B, C, T]
        }
