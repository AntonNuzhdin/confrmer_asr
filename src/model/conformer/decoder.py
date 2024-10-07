import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        model_d,
        decoder_layers,
        decoder_d,
        dropout_p,
        vocab_size,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=model_d,
            hidden_size=decoder_d,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout_p,
        )

        self.output_layer = nn.Linear(decoder_d, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)  # [batch_size, seq_length, decoder_d]
        x = self.output_layer(x)  # [batch_size, seq_length, vocab_size]
        return x
