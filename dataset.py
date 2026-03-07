"""
Multimodal Dataset
==================
PyTorch Dataset that loads preprocessed .pt files and a custom collate
function that pads variable-length OpenFace sequences within a batch.
"""

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Tuple


class MultimodalDataset(Dataset):
    """Loads preprocessed .pt sample files from disk."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

        # Load one sample to determine feature dimensions
        sample = torch.load(self.files[0], weights_only=False)
        self.audio_dim = sample["audio"].shape[0]          # 768
        self.openface_dim = sample["openface"].shape[1]    # 49
        self.transcript_dim = sample["transcript"].shape[0] # 768

        # Determine number of classes from all labels
        all_labels = []
        for f in self.files:
            s = torch.load(f, weights_only=False)
            all_labels.append(s["label"].item())
        self.label_min = min(all_labels)
        self.num_classes = max(all_labels) - self.label_min + 1

        print(f"Dataset: {len(self.files)} samples  |  "
              f"audio_dim={self.audio_dim}  openface_dim={self.openface_dim}  "
              f"transcript_dim={self.transcript_dim}  num_classes={self.num_classes}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx], weights_only=False)
        # Shift labels to be 0-indexed
        sample["label"] = sample["label"] - self.label_min
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate that pads OpenFace sequences to the max length in the batch.

    Returns:
        audio:           (B, 768)
        openface:        (B, T_max, 49)  — zero-padded
        openface_lengths: (B,)           — original sequence lengths
        transcript:      (B, 768)
        label:           (B,)
    """
    audio = torch.stack([s["audio"] for s in batch])
    transcript = torch.stack([s["transcript"] for s in batch])
    labels = torch.stack([s["label"] for s in batch])

    # Pad OpenFace sequences
    openface_seqs = [s["openface"] for s in batch]  # list of (T_i, 49)
    openface_lengths = torch.tensor([seq.shape[0] for seq in openface_seqs], dtype=torch.long)
    openface_padded = pad_sequence(openface_seqs, batch_first=True, padding_value=0.0)  # (B, T_max, 49)

    return {
        "audio": audio,
        "openface": openface_padded,
        "openface_lengths": openface_lengths,
        "transcript": transcript,
        "label": labels,
    }
