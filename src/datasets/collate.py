import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items):
    """
    Collate and pad fields in dataset items
    """
    batch = {}
    batch["text"] = [item["text"] for item in dataset_items]
    batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    batch["spectrogram"] = pad_sequence(
        [item["spectrogram"].squeeze(0).permute(1, 0) for item in dataset_items],
        batch_first=True,
    )  # [batch_size, freq, max_time]

    batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[2] for item in dataset_items]
    )

    batch["text_encoded"] = pad_sequence(
        [item["text_encoded"].squeeze(0) for item in dataset_items], batch_first=True
    )

    batch["text_encoded_length"] = torch.tensor(
        [item["text_encoded"].shape[1] for item in dataset_items]
    )

    batch["audio"] = pad_sequence(
        [item["audio"].squeeze(0) for item in dataset_items], batch_first=True
    )
    return batch
