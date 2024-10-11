import torch
import torch.nn as nn
from torch_audiomentations import (
    AddColoredNoise,
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
)


class PitchShiftAug(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super().__init__()
        self.augmentator = PitchShift(sample_rate=sample_rate, *args, **kwargs)

    def __call__(self, x):
        return self.augmentator(x.unsqueeze(1)).squeeze(1)


class ColorNoiseAug(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super().__init__()
        self.augmentator = AddColoredNoise(sample_rate=sample_rate, *args, **kwargs)

    def __call__(self, x):
        return self.augmentator(x.unsqueeze(1)).squeeze(1)


class HighPassFilterAug(nn.Module):
    def __init__(self, sample_rate, min_cutoff_freq, max_cutoff_freq, *args, **kwargs):
        super().__init__()
        self.augmentator = HighPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            *args,
            **kwargs
        )

    def __call__(self, x):
        return self.augmentator(x.unsqueeze(1)).squeeze(1)


class LowPassFilterAug(nn.Module):
    def __init__(self, sample_rate, min_cutoff_freq, max_cutoff_freq, *args, **kwargs):
        super().__init__()
        self.augmentator = LowPassFilter(
            sample_rate=sample_rate,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            *args,
            **kwargs
        )

    def __call__(self, x):
        return self.augmentator(x.unsqueeze(1)).squeeze(1)
