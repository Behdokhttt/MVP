import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


@dataclass
class OpenFaceConfig:
    feature_extraction_exe: Path


DEFAULT_OPENFACE_FEATURE_EXTRACTION = Path(
    r"C:\Users\kiafa\PycharmProjects\MVP\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
)


def _default_feature_extraction_path() -> Optional[Path]:
    env = os.environ.get("OPENFACE_FEATURE_EXTRACTION")
    if env:
        p = Path(env)
        if p.exists():
            return p

    if DEFAULT_OPENFACE_FEATURE_EXTRACTION.exists():
        return DEFAULT_OPENFACE_FEATURE_EXTRACTION

    candidates = [
        Path("C:/OpenFace/FeatureExtraction.exe"),
        Path("C:/OpenFace/build/bin/FeatureExtraction.exe"),
        Path("C:/Program Files/OpenFace/FeatureExtraction.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_openface_feature_extraction(
    config: OpenFaceConfig,
    video_path: Path,
    raw_out_dir: Path,
    stem: str,
) -> Path:
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        str(config.feature_extraction_exe),
        "-f",
        str(video_path),
        "-out_dir",
        str(raw_out_dir),
        "-of",
        stem,
        "-aus",
        "-pose",
        "-gaze",
        "-2Dfp",
        "-3Dfp",
        "-pdmparams",
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenFace FeatureExtraction failed. "
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )

    csv_path = raw_out_dir / f"{stem}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"OpenFace did not produce expected CSV: {csv_path}")

    return csv_path


def _summarize_frame_features(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=["number"])
    summary: Dict[str, Any] = {
        "n_frames": int(len(df)),
        "n_numeric_cols": int(numeric.shape[1]),
        "numeric_mean": numeric.mean(skipna=True).to_dict(),
        "numeric_std": numeric.std(skipna=True).to_dict(),
        "numeric_min": numeric.min(skipna=True).to_dict(),
        "numeric_max": numeric.max(skipna=True).to_dict(),
    }
    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        summary["confidence_mean"] = float(conf.mean(skipna=True))
        summary["confidence_min"] = float(conf.min(skipna=True))
        summary["confidence_frames_below_0_8"] = int((conf < 0.8).sum())
    if "success" in df.columns:
        succ = pd.to_numeric(df["success"], errors="coerce")
        summary["success_ratio"] = float(succ.mean(skipna=True))
    return summary


def process_video(
    video_path: Path,
    config: OpenFaceConfig,
    raw_root: Path,
    features_root: Path,
) -> Path:
    stem = video_path.stem
    raw_out_dir = raw_root / stem
    features_root.mkdir(parents=True, exist_ok=True)

    csv_path = _run_openface_feature_extraction(config, video_path, raw_out_dir, stem)

    df = pd.read_csv(csv_path)
    summary = _summarize_frame_features(df)

    payload: Dict[str, Any] = {
        "video_path": str(video_path),
        "openface_csv": str(csv_path),
        "frame_features": df,
        "summary": summary,
    }

    out_pkl = features_root / f"{stem}_openface.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    return out_pkl


def main():
    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "data" / "input_videos"
    raw_root = project_root / "data" / "output" / "openface" / "raw"
    features_root = project_root / "data" / "output" / "face_features"

    feature_exe = _default_feature_extraction_path()
    if feature_exe is None:
        raise FileNotFoundError(
            "Could not find OpenFace FeatureExtraction executable. "
            "Expected at the default location or via OPENFACE_FEATURE_EXTRACTION env var."
        )

    config = OpenFaceConfig(feature_extraction_exe=feature_exe)

    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"):
        video_files.extend(input_dir.glob(ext))

    if not video_files:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos")

    for vp in sorted(video_files):
        try:
            out_pkl = process_video(vp, config, raw_root, features_root)
            print(f"Saved OpenFace features: {out_pkl}")
        except Exception as e:
            print(f"Failed on {vp}: {e}")


if __name__ == "__main__":
    main()
