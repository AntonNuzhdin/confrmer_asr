import torch.nn as nn

from src.model.conformer.attention import RelativeMultiHeadAttention
from src.model.conformer.feedforward import FeedForward
from src.model.conformer.positional_encoding import RelativePositionalEncoder


class ConvModule(nn.Module):
    def __init__(self, model_dim, kernel_size, dropout_p):
        super().__init__()
        self.model_dim = model_dim
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(model_dim)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(
                in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1
            ),  # pointwise conv
            nn.GELU(),
            nn.Conv1d(
                in_channels=model_dim * 2,
                out_channels=model_dim * 2,
                kernel_size=kernel_size,
                groups=model_dim * 2,
                padding="same",
            ),  # depthwise conv
            nn.BatchNorm1d(model_dim * 2),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=model_dim * 2, out_channels=model_dim, kernel_size=1
            ),  # pointwise conv
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.conv_seq(x.transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, model_dim, n_heads, kernel_size, dropout_p=0.1):
        super().__init__()
        self.feed_forward1 = FeedForward(model_dim=model_dim, dropout_prob=dropout_p)
        self.attention = RelativeMultiHeadAttention(
            d_model=model_dim, num_heads=n_heads
        )
        self.conv_module = ConvModule(
            model_dim=model_dim, kernel_size=kernel_size, dropout_p=dropout_p
        )
        self.feed_forward2 = FeedForward(model_dim=model_dim, dropout_prob=dropout_p)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_enc = RelativePositionalEncoder(model_dim)

    def forward(self, x, mask):
        x = 0.5 * self.feed_forward1(x) + x
        x = self.attention(x, self.positional_enc(x), mask) + x
        x = self.conv_module(x) + x
        x = 0.5 * self.feed_forward2(x) + x
        x = self.layer_norm(x)
        return x
