"""
Multimodal Emotion Classification Network
==========================================

Architecture:
  1. AudioProcessor   — FC layers on MFCC features        (130 → hidden)
  2. OpenFaceProcessor — GRU on sequential frame features  (22 per frame → hidden)
  3. TranscriptProcessor — FC layers on BERT embedding     (768 → hidden)
  4. Combiner         — concatenate all three → FC → num_classes
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AudioProcessor(nn.Module):
    """Fully connected network for MFCC audio features."""

    def __init__(self, input_dim: int = 130, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 130)
        Returns:
            (B, output_dim)
        """
        return self.net(x)


class OpenFaceProcessor(nn.Module):
    """Bidirectional GRU for sequential OpenFace features."""

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional doubles the hidden dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, T_max, 22)  — zero-padded sequences
            lengths: (B,)            — original lengths
        Returns:
            (B, output_dim)
        """
        # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
        lengths_clamped = lengths.clamp(min=1).cpu()

        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        # hidden: (num_layers * 2, B, hidden_dim) — take last layer forward + backward
        # Forward final hidden state
        h_fwd = hidden[-2]  # (B, hidden_dim)
        h_bwd = hidden[-1]  # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden_dim * 2)
        return self.fc(h_cat)


class TranscriptProcessor(nn.Module):
    """Fully connected network for BERT transcript embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 768)
        Returns:
            (B, output_dim)
        """
        return self.net(x)


class MultimodalEmotionClassifier(nn.Module):
    """
    Combines audio, openface, and transcript processors
    via late fusion (concatenation) followed by classification head.
    """

    def __init__(
        self,
        num_classes: int,
        audio_input_dim: int = 130,
        openface_input_dim: int = 22,
        transcript_input_dim: int = 768,
        branch_hidden_dim: int = 256,
        branch_output_dim: int = 128,
        combiner_hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_processor = AudioProcessor(
            input_dim=audio_input_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=branch_output_dim,
            dropout=dropout,
        )
        self.openface_processor = OpenFaceProcessor(
            input_dim=openface_input_dim,
            hidden_dim=rnn_hidden_dim,
            output_dim=branch_output_dim,
            num_layers=rnn_num_layers,
            dropout=dropout,
        )
        self.transcript_processor = TranscriptProcessor(
            input_dim=transcript_input_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=branch_output_dim,
            dropout=dropout,
        )

        # Combiner: takes concatenated branch outputs → classification
        fused_dim = branch_output_dim * 3  # audio + openface + transcript
        self.combiner = nn.Sequential(
            nn.Linear(fused_dim, combiner_hidden_dim),
            nn.BatchNorm1d(combiner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combiner_hidden_dim, num_classes),
        )

    def forward(
        self,
        audio: torch.Tensor,
        openface: torch.Tensor,
        openface_lengths: torch.Tensor,
        transcript: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio:            (B, 130)
            openface:         (B, T_max, 22)
            openface_lengths: (B,)
            transcript:       (B, 768)
        Returns:
            logits: (B, num_classes)
        """
        audio_out = self.audio_processor(audio)            # (B, branch_output_dim)
        openface_out = self.openface_processor(openface, openface_lengths)  # (B, branch_output_dim)
        transcript_out = self.transcript_processor(transcript)  # (B, branch_output_dim)

        fused = torch.cat([audio_out, openface_out, transcript_out], dim=1)  # (B, 3 * branch_output_dim)
        logits = self.combiner(fused)  # (B, num_classes)
        return logits
