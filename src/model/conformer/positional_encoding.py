import torch
import torch.nn as nn


# from DL-1 seminars
class RelativePositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_len=8000):
        super().__init__()
        self.model_dim = model_dim

        positional_encodings = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / model_dim)
        )

        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x):
        return x + self.positional_encodings[:, : x.size(1)]
