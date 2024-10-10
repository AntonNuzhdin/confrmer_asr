import torch.nn as nn
import torch.nn.functional as F


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )

    def forward(self, x, seq_lengths):
        x = x.transpose(1, 2)
        x = F.relu(self.batch_norm(x))
        x = x.transpose(1, 2)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x_packed, _ = self.rnn(x_packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)

        x = x[:, :, : self.hidden_size] + x[:, :, self.hidden_size :]

        return x
