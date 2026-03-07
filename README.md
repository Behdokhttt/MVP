# Multimodal Emotion Classification System

A comprehensive system for emotion classification from short video clips (3-15 seconds) that combines audio, speech transcription, and facial visual cues.

## Features

### Audio Processing
- **Traditional Features**: MFCCs, spectral features, energy features, pitch features
- **Deep Learning Embeddings**: Wav2Vec2 pre-trained embeddings
- **OpenSMILE Integration**: eGeMAPSv02 feature set for comprehensive audio analysis
- **Statistical Functionals**: Mean, std, min, max, skew, kurtosis over temporal windows

### Video Processing
- **Batch Processing**: Automatically processes all videos in `data/input_videos/`
- **Audio Extraction**: Extracts audio from video files using moviepy
- **Feature Storage**: Saves audio features as pickle files in `data/output/audio/`
- **Speech Transcription**: Transcribes speech using Whisper and saves as text files in `data/output/transcripts/`

### Facial Visual Cues
- **OpenFace 2.0**: Extracts facial cues (AUs, head pose, gaze, landmarks) and saves features for reuse in emotion recognition

### Multimodal Architecture
- Audio feature extraction pipeline (implemented)
- Speech transcription features (implemented)
- Facial visual cue extraction (implemented)
- Multimodal fusion and classification (implemented)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Install spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Install FFmpeg** (required for Whisper transcription):
   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`

5. **Install OpenFace 2.0** (required for facial feature extraction):
   - Download/build OpenFace 2.0 and locate `FeatureExtraction.exe` (Windows)
   - Set an environment variable pointing to it:
     ```bash
     setx OPENFACE_FEATURE_EXTRACTION "C:\\path\\to\\FeatureExtraction.exe"
     ```
   - Restart your terminal/IDE after setting environment variables

**Note**: If FFmpeg is not installed, the system will automatically fall back to using librosa for audio processing.

## Usage

### Process Videos and Extract Audio Features
```bash
python process_videos.py
```

### Transcribe Speech from Videos
```bash
python transcribe_videos.py
```

This will:
- Read all video files from `data/input_videos/`
- Extract audio from each video
- Transcribe speech using Whisper (with SpeechRecognition fallback)
- Save transcripts as text files in `data/output/transcripts/`

### Fine-tune Emotion Text Classifier
```bash
python emotion_text_classifier.py
```

This will:
- Initialize DeBERTa-v3 model for emotion classification
- Provide framework for fine-tuning on labeled emotion data
- Extract emotion-specific features with confidence scores

### Extract Features from Transcripts
```bash
python transcript_features.py
```

This will:
- Read all transcript files from `data/output/transcripts/`
- Extract linguistic, sentiment, and semantic features
- Use fine-tuned DeBERTa-v3 if available (fallback to frozen embeddings)
- Save features as pickle files in `data/output/transcript_features/`

### Extract Facial Features (OpenFace)
```bash
python extract_openface_features.py
```

This will:
- Read all video files from `data/input_videos/`
- Run OpenFace 2.0 `FeatureExtraction` to produce per-frame features
- Save raw OpenFace outputs under `data/output/openface/raw/<video_stem>/`
- Save a reusable pickle per video under `data/output/face_features/<video_stem>_openface.pkl`

The script looks for `FeatureExtraction.exe` at the default location:
`C:\Users\kiafa\PycharmProjects\MVP\OpenFace_2.2.0_win_x64\FeatureExtraction.exe`

If you prefer, you can override it by setting the environment variable `OPENFACE_FEATURE_EXTRACTION` to the full path of `FeatureExtraction.exe`.

### Manual Audio Feature Extraction
```python
from audio_features import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor(sample_rate=16000, use_transformer=True)

# Extract features from audio file
features = extractor.extract_all_features("audio_file.wav")
print(f"Extracted {len(features)} features")
```

### Preprocess and Train
```bash
# Step 1: Combine all modalities into per-video .pt files
python preprocess.py

# Step 2: Train the multimodal classifier
python train.py --epochs 50 --batch_size 8 --lr 1e-3
```

## Project Structure
```
MVP/
├── main.py              # Main multimodal classifier
├── audio_features.py    # Audio feature extraction pipeline
├── process_videos.py    # Batch video processing script
├── transcribe_videos.py # Speech transcription script
├── transcript_features.py # Transcript feature extraction script
├── emotion_text_classifier.py # Fine-tuned DeBERTa-v3 emotion classifier
├── extract_openface_features.py # OpenFace 2.0 facial feature extraction script
├── preprocess.py        # Combines all modalities into single .pt files per video
├── dataset.py           # PyTorch Dataset with variable-length sequence collation
├── model.py             # Multimodal network (AudioProcessor + OpenFaceProcessor + TranscriptProcessor + Combiner)
├── train.py             # Training script with train/val split and checkpointing
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── data/
    ├── input_videos/    # Input video files
    ├── output/
    │   ├── audio/       # Extracted audio features (.pkl files)
    │   ├── face_features/ # OpenFace features (.pkl files)
    │   ├── transcripts/ # Speech transcripts (.txt files)
    │   ├── transcript_features/ # Transcript features (.pkl files)
    │   └── openface/
    │       └── raw/     # Raw OpenFace per-video outputs
    └── processed/       # Combined per-video .pt files (generated by preprocess.py)
```

## Features Used for Training

### Audio Features (130-dim fixed vector per video)

The audio branch uses **MFCC (Mel-Frequency Cepstral Coefficients)** statistics
extracted from the audio signal. MFCCs capture the spectral envelope of speech,
which correlates with vocal tract shape and is highly informative for emotion.

10 summary statistics are computed over the 13 MFCC coefficients:

| Statistic | Dim | Description |
|---|---:|---|
| `mfcc_mean` | 13 | Average MFCC value across all frames |
| `mfcc_std` | 13 | Standard deviation — captures variability in vocal quality |
| `mfcc_min` | 13 | Minimum value — floor of spectral characteristics |
| `mfcc_max` | 13 | Maximum value — peak spectral characteristics |
| `mfcc_skew` | 13 | Skewness — asymmetry in MFCC distribution |
| `mfcc_kurt` | 13 | Kurtosis — peakedness of MFCC distribution |
| `delta_mfcc_mean` | 13 | Mean of 1st-order differences — rate of spectral change |
| `delta_mfcc_std` | 13 | Std of 1st-order differences — variability in spectral change |
| `delta2_mfcc_mean` | 13 | Mean of 2nd-order differences — acceleration of spectral change |
| `delta2_mfcc_std` | 13 | Std of 2nd-order differences — variability in acceleration |
| **Total** | **130** | |

### OpenFace Features (22-dim sequential, per frame)

The OpenFace branch uses a curated subset of per-frame facial features,
processed by a bidirectional GRU to capture temporal dynamics.

| Group | Columns | Dim | Description |
|---|---|---:|---|
| **Gaze angles** | `gaze_angle_x`, `gaze_angle_y` | 2 | Summarised gaze direction in radians. Camera-independent (unlike raw gaze vectors). |
| **Head rotation** | `pose_Rx`, `pose_Ry`, `pose_Rz` | 3 | Head pitch, yaw, roll in radians. Emotion-relevant (e.g. head tilt, nodding). Translation (`pose_Tx/Ty/Tz`) is dropped as it depends on camera distance/setup. |
| **AU intensities** | `AU01_r` .. `AU45_r` | 17 | Action Unit regression values (continuous 0–5 scale). Each AU corresponds to a specific facial muscle movement (e.g. AU12 = lip corner puller = smile). Binary AU classification columns (`AU_c`) are dropped as they are redundant with the regression values. |
| **Total** | | **22** | |

**Dropped features and rationale:**
- `gaze_0_x/y/z`, `gaze_1_x/y/z` (6) — raw per-eye 3D vectors; `gaze_angle_x/y` already summarises these
- `pose_Tx/Ty/Tz` (3) — head position in camera space; camera-dependent and not emotion-relevant
- `AU*_c` (18) — binary presence of AUs; redundant with continuous `AU*_r` intensity values
- Eye/face landmarks (2D + 3D), shape parameters — person-specific and noisy; AUs are the generalised representation

### Transcript Features (768-dim fixed vector per video)

BERT (or DeBERTa-v3) [CLS] mean-pooled embeddings extracted from the speech
transcript. These capture semantic meaning and are processed by an FC branch.

## Supported Video Formats
- MP4, AVI, MOV, MKV, FLV, WMV

## Output Files
Each video file generates corresponding output files:
- Input: `data/input_videos/v_001.mp4`
- Audio Features: `data/output/audio/v_001_audio_features.pkl`
- Transcript: `data/output/transcripts/v_001_transcript.txt`
- Transcript Features: `data/output/transcript_features/v_001_transcript_features.pkl`

## Emotion Labels
Seven emotion classes supported:
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise
- Neutral

## Model Architecture

The multimodal network uses **late fusion** with three parallel branches:

```
Audio (130)  ──► FC(256) → FC(128) ──────────────────────────────┐
OpenFace (T×22) ──► BiGRU(128, 2 layers) → FC(128) ─────────────┼──► Concat(384) → FC(128) → FC(num_classes)
Transcript (768) ──► FC(256) → FC(128) ──────────────────────────┘
```

All FC blocks include BatchNorm + ReLU + Dropout. The GRU uses `pack_padded_sequence` to handle variable-length video sequences within a batch.

## Next Steps
1. Collect more labeled video data
2. Evaluate performance and tune hyperparameters
3. Experiment with attention-based fusion
4. Add per-class metrics (precision, recall, F1)

## Dependencies
- librosa: Audio processing
- transformers: Wav2Vec2 and BERT models
- opensmile: eGeMAPS features
- torch: Deep learning framework
- moviepy: Video audio extraction
- openai-whisper: Speech transcription
- speechrecognition: Backup transcription
- nltk: Natural language processing
- spacy: Advanced NLP and POS tagging
- textblob: Sentiment analysis
- sentencepiece: Required for DeBERTa tokenizer
- protobuf: Required for transformers
- ffmpeg-python: FFmpeg Python bindings
- opencv-python: Video processing
- numpy, scipy: Numerical operations

**System Requirements**:
- FFmpeg: Required for Whisper audio processing (auto-fallback to librosa if not available)
- spaCy model: `en_core_web_sm` (auto-downloaded on first run)
- sentencepiece: Required for DeBERTa-v3 tokenizer
