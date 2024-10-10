import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.deepspeech2.conv_module import Conv2dModule
from src.model.deepspeech2.rnn import RNNModule


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        freq,
        vocab_size,
        kernel_size1=(41, 11),
        kernel_size2=(21, 11),
        kernel_size3=(21, 11),
        hidden_size=512,
        dropout=0.0,
        rnn_layers=5,
    ):
        super().__init__()
        self.conv2dModule = Conv2dModule(kernel_size1, kernel_size2, kernel_size3)

        freq_size = freq

        padding1 = self.conv2dModule._get_padding_size(kernel_size1)[0]
        freq_size = self._get_dim_after_conv2d(
            freq_size, kernel_size1[0], stride=2, padding=padding1
        )

        padding2 = self.conv2dModule._get_padding_size(kernel_size2)[0]
        freq_size = self._get_dim_after_conv2d(
            freq_size, kernel_size2[0], stride=2, padding=padding2
        )

        padding3 = self.conv2dModule._get_padding_size(kernel_size3)[0]
        freq_size = self._get_dim_after_conv2d(
            freq_size, kernel_size3[0], stride=2, padding=padding3
        )

        input_size = freq_size * 96

        self.rnn = nn.ModuleList()
        self.rnn.append(
            RNNModule(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        )

        for _ in range(rnn_layers - 1):
            self.rnn.append(
                RNNModule(
                    input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
                )
            )

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.permute(0, 2, 1)
        spec_length = spectrogram_length
        x = x.unsqueeze(1)
        x = self.conv2dModule(x)
        spec_length = self._get_seq_lens(spec_length)
        x = self._apply_mask(x, spec_length)
        batch_size, channels, freq_size, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * freq_size)

        for rnn_block in self.rnn:
            x = rnn_block(x, seq_lengths=spec_length)

        x = self.linear(x)

        log_probs = F.log_softmax(x, dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": spec_length,
        }

    def _get_seq_lens(self, seq_lengths):
        for m in self.conv2dModule.conv2d_module:
            if isinstance(m, nn.Conv2d):
                kernel, stride, padding = m.kernel_size[1], m.stride[1], m.padding[1]
                seq_lengths = ((seq_lengths + 2 * padding - kernel) // stride) + 1
        return seq_lengths.int()

    def _apply_mask(self, x, lengths):
        bs, _, _, time = x.size()
        mask = torch.arange(time, device=x.device).view(1, 1, 1, time) >= lengths.view(
            bs, 1, 1, 1
        )
        x = x.masked_fill(mask, 0)
        return x

    @staticmethod
    def _get_dim_after_conv2d(features_in, kernel_size, stride, padding):
        return ((features_in + 2 * padding - (kernel_size - 1) - 1) // stride) + 1
