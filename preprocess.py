"""
Preprocessing Pipeline
======================
Reads extracted features from all modalities (audio, openface, transcript)
and combines them into a single .pt file per video for efficient training.

Audio:     MFCC statistics (130-dim) from audio_features.pkl
OpenFace:  sequential frame-level features (gaze, pose, AUs) from openface CSV
Transcript: BERT embeddings (768-dim) from transcript_features.pkl
Labels:    integer class labels from labels.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


# ── Feature column selections for OpenFace ──────────────────────────────────
# Gaze angles (2) — summarised gaze direction, camera-independent
OPENFACE_GAZE_COLS = [
    'gaze_angle_x', 'gaze_angle_y',
]

# Head rotation only (3) — emotion-relevant; translation dropped (camera-dependent)
OPENFACE_POSE_COLS = [
    'pose_Rx', 'pose_Ry', 'pose_Rz',
]

# AU regression intensities (17) — continuous 0-5 scale; binary AU_c dropped (redundant)
OPENFACE_AU_R_COLS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
]

OPENFACE_FEATURE_COLS = (
    OPENFACE_GAZE_COLS + OPENFACE_POSE_COLS + OPENFACE_AU_R_COLS
)  # 22 features per frame


class MultimodalPreprocessor:
    """Reads raw extracted features and saves unified .pt files per video."""

    def __init__(
        self,
        audio_dir: str = "data/output/audio",
        openface_dir: str = "data/output/openface/raw",
        transcript_feat_dir: str = "data/output/transcript_features",
        labels_path: str = "data/output/labels.csv",
        output_dir: str = "data/processed",
    ):
        self.audio_dir = Path(audio_dir)
        self.openface_dir = Path(openface_dir)
        self.transcript_feat_dir = Path(transcript_feat_dir)
        self.labels_path = Path(labels_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loaders ──────────────────────────────────────────────────────────

    def load_labels(self) -> Dict[str, int]:
        """Load video_id -> label mapping from labels.csv."""
        labels = {}
        with open(self.labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                video_id = parts[0].strip()
                label = int(parts[1].strip())
                labels[video_id] = label
        print(f"Loaded {len(labels)} labels")
        return labels

    def load_mfcc_features(self, video_id: str) -> Optional[np.ndarray]:
        """Extract MFCC statistics (130-dim) from the audio pickle.

        Concatenates 10 summary statistics of 13 MFCCs:
          mfcc_mean, mfcc_std, mfcc_min, mfcc_max, mfcc_skew, mfcc_kurt,
          delta_mfcc_mean, delta_mfcc_std, delta2_mfcc_mean, delta2_mfcc_std
        """
        pkl_path = self.audio_dir / f"{video_id}_audio_features.pkl"
        if not pkl_path.exists():
            print(f"  [WARN] Audio features not found: {pkl_path}")
            return None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        MFCC_KEYS = [
            'mfcc_mean', 'mfcc_std', 'mfcc_min', 'mfcc_max',
            'mfcc_skew', 'mfcc_kurt',
            'delta_mfcc_mean', 'delta_mfcc_std',
            'delta2_mfcc_mean', 'delta2_mfcc_std',
        ]
        parts = []
        for key in MFCC_KEYS:
            if key not in data:
                print(f"  [WARN] Missing MFCC key '{key}' for {video_id}")
                return None
            parts.append(np.asarray(data[key], dtype=np.float32))

        return np.concatenate(parts)  # (130,)

    def load_openface_features(self, video_id: str) -> Optional[np.ndarray]:
        """Load sequential OpenFace features (T, 22) from the CSV."""
        csv_path = self.openface_dir / video_id / f"{video_id}.csv"
        if not csv_path.exists():
            print(f"  [WARN] OpenFace CSV not found: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Select only the feature columns we care about
        missing = [c for c in OPENFACE_FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"  [WARN] Missing OpenFace columns: {missing}")
            return None

        features = df[OPENFACE_FEATURE_COLS].values.astype(np.float32)

        # Filter out frames where face detection failed (confidence < 0.5)
        if "confidence" in df.columns:
            mask = df["confidence"].values >= 0.5
            features = features[mask]

        if len(features) == 0:
            print(f"  [WARN] No valid OpenFace frames for {video_id}")
            return None

        return features  # shape: (T, 49)

    def load_bert_features(self, video_id: str) -> Optional[np.ndarray]:
        """Extract the 768-dim BERT embedding from transcript_features pickle."""
        pkl_path = self.transcript_feat_dir / f"{video_id}_transcript_features.pkl"
        if not pkl_path.exists():
            print(f"  [WARN] Transcript features not found: {pkl_path}")
            return None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        emb = np.array(
            [float(data[f"bert_emb_{i}"]) for i in range(768)],
            dtype=np.float32,
        )
        return emb

    # ── Main processing ──────────────────────────────────────────────────

    def process_video(self, video_id: str, label: int) -> bool:
        """Process a single video: load all modalities and save as .pt."""
        audio_feat = self.load_mfcc_features(video_id)
        openface_feat = self.load_openface_features(video_id)
        bert_feat = self.load_bert_features(video_id)

        if audio_feat is None or openface_feat is None or bert_feat is None:
            print(f"  [SKIP] {video_id} — missing modality")
            return False

        sample = {
            "video_id": video_id,
            "audio": torch.tensor(audio_feat, dtype=torch.float32),       # (130,)
            "openface": torch.tensor(openface_feat, dtype=torch.float32), # (T, 22)
            "transcript": torch.tensor(bert_feat, dtype=torch.float32),   # (768,)
            "label": torch.tensor(label, dtype=torch.long),
        }

        out_path = self.output_dir / f"{video_id}.pt"
        torch.save(sample, out_path)
        T = openface_feat.shape[0]
        print(f"  [OK] {video_id}  audio={audio_feat.shape}  openface=({T}, {openface_feat.shape[1]})  transcript={bert_feat.shape}  label={label}")
        return True

    def run(self) -> None:
        """Process all videos listed in labels.csv."""
        labels = self.load_labels()

        print(f"\nProcessing {len(labels)} videos...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)

        success, fail = 0, 0
        for video_id, label in labels.items():
            if self.process_video(video_id, label):
                success += 1
            else:
                fail += 1

        print("-" * 60)
        print(f"Done.  Success: {success}  |  Failed/Skipped: {fail}")
        print(f"Processed files saved to: {self.output_dir}")


if __name__ == "__main__":
    preprocessor = MultimodalPreprocessor()
    preprocessor.run()
