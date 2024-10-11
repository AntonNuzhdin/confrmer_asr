import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class FrequencyMaskingAug(nn.Module):
    def __init__(self, freq_mask_param, p):
        super().__init__()
        self.p = p
        self.augmentator = FrequencyMasking(freq_mask_param)

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return self.augmentator(x)
        return x


class TimeMaskingAug(nn.Module):
    def __init__(self, time_mask_param, p):
        super().__init__()
        self.augmentator = TimeMasking(time_mask_param)
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return self.augmentator(x)
        return x
