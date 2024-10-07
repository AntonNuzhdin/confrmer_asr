import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dim,
        dropout_prob,
    ):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, x):
        return self.feedforward(x)
