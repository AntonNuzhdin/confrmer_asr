import torch.nn as nn
import torch.nn.functional as F


class Conv2dModule(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, kernel_size3):
        super().__init__()

        self.conv2d_module = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=kernel_size1,
                padding=self._get_padding_size(kernel_size1),
                stride=(2, 2),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=kernel_size2,
                padding=self._get_padding_size(kernel_size2),
                stride=(2, 2),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=kernel_size3,
                padding=self._get_padding_size(kernel_size3),
                stride=(2, 1),
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv2d_module(x)

    @staticmethod
    def _get_padding_size(kernel_size):
        return (kernel_size[0] // 2, kernel_size[1] // 2)
