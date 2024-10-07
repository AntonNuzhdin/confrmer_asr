import torch.nn as nn


class ConvolutionSubsampling(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=out_channels, kernel_size=3, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        # x - [bs, freq, time]
        x = self.subsampling(x.unsqueeze(1))
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(x.size(0), x.size(1), -1)
