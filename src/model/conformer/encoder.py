import torch
import torch.nn as nn

from src.model.conformer.conformer_block import ConformerBlock
from src.model.conformer.subsampling import ConvolutionSubsampling


class Encoder(nn.Module):
    def __init__(
        self,
        freq_dim,
        model_dim,
        encoder_layers,
        n_heads,
        kernel_size,
        dropout_p=0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            ConvolutionSubsampling(out_channels=model_dim),
            nn.Linear(
                in_features=model_dim
                * self._get_dim_after_conv2d(
                    self._get_dim_after_conv2d(freq_dim, 3, 2), 3, 2
                ),
                out_features=model_dim,
            ),
            nn.Dropout(p=dropout_p),
        )

        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                    dropout_p=dropout_p,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, x):
        # [batch_size, freq, max_time]
        x = self.proj(x)
        for conf_block in self.conformer_blocks:
            x = conf_block(x)
        return x

    @staticmethod
    def _get_dim_after_conv2d(features_in, kernel_size, stride):
        # (w-(k-1)-1) / s + 1
        return ((features_in - (kernel_size - 1) - 1) // stride) + 1
