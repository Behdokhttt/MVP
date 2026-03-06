# Video Speaker Splitting System

A comprehensive system that automatically splits videos of two people talking into sections based on turn-taking. The system identifies who is speaking at each moment and creates video sections with overlap to ensure complete capture of each person's speech.

## Features

### Speaker Diarization
- **Advanced Speaker Recognition**: Uses PyAnnote audio for accurate speaker identification
- **Fallback Methods**: Energy-based diarization when advanced models aren't available
- **Speech Activity Detection**: Whisper-based speech detection with word-level timestamps
- **Multi-Speaker Support**: Configurable for 2+ speakers (optimized for 2)

### Video Processing
- **Turn-Based Splitting**: Automatically detects speaker changes and creates sections
- **Overlap Handling**: Creates overlapping sections when speakers change to capture complete speech
- **Format Support**: MP4, AVI, MOV, MKV, FLV, WMV
- **Quality Preservation**: Maintains original video quality with H.264 encoding

### Output Features
- **Named Sections**: Files named as `{original_name}_section_{number}_{speaker}.mp4`
- **Timing Information**: JSON file with detailed speaker timing data
- **Batch Processing**: Handles multiple videos automatically
- **Progress Tracking**: Detailed console output for monitoring

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install system dependencies (see INSTALLATION.md for details)
# - FFmpeg
# - spaCy model
# - (Optional) Hugging Face token for best performance
```

### 2. Prepare Videos
```bash
# Create input directory
mkdir -p data/input_videos

# Place your videos in the directory
# Supported formats: MP4, AVI, MOV, MKV, FLV, WMV
```

### 3. Run the System
```bash
python split_video_by_speaker.py
```

### 4. Check Results
```bash
# Output will be in data/output/split_videos/
ls data/output/split_videos/
```

## Example Output

For a video named `conversation.mp4` with two speakers:

```
data/output/split_videos/
├── conversation_section_000_SPEAKER_00.mp4    # First person speaks
├── conversation_section_001_SPEAKER_01.mp4    # Second person responds
├── conversation_section_002_SPEAKER_00.mp4    # First person continues
├── conversation_section_003_SPEAKER_01.mp4    # Second person responds
└── conversation_segments.json                # Timing information
```

## Advanced Usage

### Custom Overlap Duration
```python
from split_video_by_speaker import VideoSplitter

# Create splitter with custom overlap (default: 0.5 seconds)
splitter = VideoSplitter()
splitter.process_all_videos(overlap_duration=1.0)  # 1 second overlap
```

### Process Single Video
```python
splitter = VideoSplitter()
success = splitter.process_single_video("path/to/video.mp4", overlap_duration=0.3)
```

### Custom Directories
```python
splitter = VideoSplitter(
    input_dir="my_videos",
    output_dir="my_output"
)
splitter.process_all_videos()
```

## Technical Details

### Speaker Identification Process

1. **Audio Extraction**: Extracts 16kHz audio from video
2. **Speech Detection**: Uses Whisper to identify speech segments
3. **Speaker Diarization**: PyAnnote identifies who speaks when
4. **Segment Combination**: Merges speech and speaker information
5. **Turn Detection**: Identifies speaker changes and creates sections
6. **Video Splitting**: Extracts video sections with overlap

### Overlap Handling

When speakers change, the system creates overlapping sections:
- Previous speaker's section extends into the next speaker's start
- Overlap duration is configurable (default: 0.5 seconds)
- Ensures no speech is cut off at speaker transitions

### Fallback Methods

If PyAnnote models aren't available:
- Energy-based voice activity detection
- Simple speaker alternation (every 3 seconds)
- Still provides basic speaker separation

## Performance

### Processing Speed
- **CPU**: ~0.5x real-time (10-minute video takes ~20 minutes)
- **GPU (CUDA)**: ~2-3x real-time (10-minute video takes ~3-5 minutes)

### Memory Requirements
- **Base**: 4GB RAM for short videos (<5 minutes)
- **Recommended**: 8GB+ RAM for longer videos
- **GPU**: Additional 4GB VRAM for CUDA acceleration

### Disk Space
- Input: Original video size
- Output: ~1.2x original size (due to overlap sections)
- Temporary: ~2x original size during processing

## Troubleshooting

### Common Issues

**No speech detected:**
- Check audio quality and volume
- Ensure video has clear speech
- Try different language settings in the code

**Poor speaker separation:**
- Ensure distinct speaker voices
- Check for background noise
- Consider longer overlap duration

**Memory errors:**
- Process shorter videos
- Close other applications
- Use CPU version if GPU memory is limited

**Permission errors:**
- Check file/directory permissions
- Run as administrator if needed
- Ensure output directory is writable

## File Structure

```
MVP/
├── split_video_by_speaker.py    # Main splitting script
├── requirements.txt              # Python dependencies
├── INSTALLATION.md               # Detailed installation guide
├── README_SPEAKER_SPLITTING.md   # This file
└── data/
    ├── input_videos/             # Place videos here
    └── output/
        └── split_videos/         # Results appear here
            ├── video_section_000_SPEAKER_00.mp4
            ├── video_section_001_SPEAKER_01.mp4
            └── video_segments.json
```

## Dependencies

### Core Libraries
- **torch**: Deep learning framework
- **whisper**: Speech transcription
- **pyannote.audio**: Speaker diarization
- **moviepy**: Video processing
- **librosa**: Audio analysis

### Supporting Libraries
- **numpy**, **scipy**: Numerical operations
- **soundfile**: Audio file handling
- **transformers**: Model loading
- **opencv-python**: Video I/O

### System Requirements
- **FFmpeg**: Audio/video processing
- **Python 3.8+**: Runtime environment
- **CUDA (optional)**: GPU acceleration

## Configuration

### Environment Variables
```bash
# Hugging Face token for PyAnnote models (optional)
export HF_TOKEN="your_token_here"

# CUDA device selection (optional)
export CUDA_VISIBLE_DEVICES="0"
```

### Script Parameters
```python
# In split_video_by_speaker.py

# Number of expected speakers
NUM_SPEAKERS = 2

# Default overlap duration (seconds)
DEFAULT_OVERLAP = 0.5

# Audio sample rate
SAMPLE_RATE = 16000

# Video codec
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'
```

## API Reference

### VideoSplitter Class

```python
class VideoSplitter:
    def __init__(self, input_dir="data/input_videos", 
                 output_dir="data/output/split_videos")
    
    def process_all_videos(self, overlap_duration=0.5)
    def process_single_video(self, video_path, overlap_duration=0.5)
    def create_turn_sections(self, segments, overlap_duration=0.5)
    def split_video_section(self, video_path, start_time, end_time, output_path)
```

### SpeakerDiarizer Class

```python
class SpeakerDiarizer:
    def __init__(self, num_speakers=2)
    def extract_audio(self, video_path)
    def detect_speech_activity(self, audio_path)
    def perform_diarization(self, audio_path)
    def combine_speech_and_speakers(self, speech_segments, speaker_segments)
```

## License

This project is provided as-is for educational and research purposes. Please ensure you have the right to process and split any videos you use with this system.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the installation guide
3. Verify your setup matches requirements
4. Create an issue with detailed information
