import torch.nn as nn
import torch.nn.functional as F

from src.model.conformer.decoder import Decoder
from src.model.conformer.encoder import Encoder


class Conformer(nn.Module):
    def __init__(
        self,
        freq_dim,
        vocab_size,
        encoder_layers=16,
        decoder_layers=1,
        encoder_d=144,
        decoder_d=320,
        dropout_p=0.1,
        n_heads=4,
        kernel_size=32,
    ):
        super().__init__()

        self.encoder = Encoder(
            freq_dim=freq_dim,
            model_dim=encoder_d,
            dropout_p=dropout_p,
            encoder_layers=encoder_layers,
            n_heads=n_heads,
            kernel_size=kernel_size,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            model_d=encoder_d,
            decoder_layers=decoder_layers,
            decoder_d=decoder_d,
            dropout_p=dropout_p,
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.decoder(self.encoder(spectrogram, spectrogram_length))
        log_probs = F.log_softmax(x, dim=-1)
        log_probs_length = self._transform_input_lengths(spectrogram_length)
        return {
            "log_probs": log_probs,
            "log_probs_length": log_probs_length,
        }

    def _transform_input_lengths(self, input_lengths):
        return ((input_lengths - 1) // 2 - 1) // 2

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
