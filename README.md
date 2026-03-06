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
- Multimodal fusion and classification (planned)

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

### Multimodal Emotion Classification
```python
from main import EmotionClassifier

# Initialize classifier
classifier = EmotionClassifier()

# Process video clip
features = classifier.process_video_clip("video_file.mp4")
emotion, confidence = classifier.predict_emotion(features)
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
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
└── data/
    ├── input_videos/   # Input video files
    └── output/
        ├── audio/      # Extracted audio features (.pkl files)
        ├── face_features/ # OpenFace features (.pkl files)
        ├── transcripts/ # Speech transcripts (.txt files)
        ├── transcript_features/ # Transcript features (.pkl files)
        └── openface/
            └── raw/     # Raw OpenFace per-video outputs
```

## Audio Features Extracted

### Low-level Descriptors
- **MFCCs**: 13 coefficients + delta + delta-delta
- **Spectral**: centroid, bandwidth, rolloff, flux
- **Energy**: RMS energy statistics
- **Pitch**: fundamental frequency, voicing ratio

### Deep Learning Features
- **Wav2Vec2**: Pre-trained transformer embeddings
- **OpenSMILE**: eGeMAPSv02 feature set (88 features)

### Statistical Functionals
For each temporal feature:
- Mean, standard deviation
- Minimum, maximum
- Skewness, kurtosis

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

## Next Steps
1. Implement speech transcription features
2. Add facial visual cue extraction
3. Design multimodal fusion architecture
4. Train classification models
5. Evaluate performance on emotion dataset

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
