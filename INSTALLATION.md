# Installation Guide for Video Speaker Splitting System

This guide provides step-by-step instructions for installing all required packages and dependencies for the video speaker splitting system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MVP
```

## Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Install Additional System Dependencies

### FFmpeg (Required for audio/video processing)

**Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract the files to a location (e.g., `C:\ffmpeg`)
3. Add the bin directory to your PATH environment variable:
   - Go to System Properties → Advanced → Environment Variables
   - Add `C:\ffmpeg\bin` to the PATH variable
4. Verify installation:
   ```bash
   ffmpeg -version
   ```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Linux (Fedora/CentOS):**
```bash
sudo yum install ffmpeg
```

### spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## Step 5: Install PyTorch (Optional - for GPU support)

If you have an NVIDIA GPU with CUDA support, you can install the GPU version of PyTorch for better performance:

```bash
# Check CUDA version first
nvidia-smi

# Install appropriate PyTorch version (example for CUDA 11.8)
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

If you don't have a GPU, the CPU version installed in Step 3 will work fine.

## Step 6: Configure Hugging Face Access (Optional)

For the best speaker diarization performance, you may want to use the official PyAnnote models:

1. Create a Hugging Face account at [https://huggingface.co](https://huggingface.co)
2. Go to [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the user agreement
4. Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Set the token as an environment variable:
   ```bash
   # Windows:
   setx HF_TOKEN "your_token_here"
   
   # macOS/Linux:
   export HF_TOKEN="your_token_here"
   ```

## Step 7: Verify Installation

Run the following command to test the installation:

```bash
python -c "import torch; import whisper; import pyannote.audio; import moviepy; print('All packages installed successfully!')"
```

## Step 8: Prepare Input Directory

Create the input directory for your videos:

```bash
mkdir -p data/input_videos
```

Place your video files (MP4, AVI, MOV, MKV, FLV, WMV) in the `data/input_videos/` directory.

## Usage

Once everything is installed, you can run the video splitting system:

```bash
python split_video_by_speaker.py
```

This will:
1. Process all videos in `data/input_videos/`
2. Identify speakers and their speaking times
3. Split videos into sections based on turn-taking
4. Save output to `data/output/split_videos/`

## Output Structure

```
data/output/split_videos/
├── original_video_section_000_SPEAKER_00.mp4
├── original_video_section_001_SPEAKER_01.mp4
├── original_video_section_002_SPEAKER_00.mp4
└── original_video_segments.json
```

- Video sections are named with format: `{original_name}_section_{number}_{speaker}.mp4`
- The JSON file contains detailed timing and speaker information

## Troubleshooting

### Common Issues

1. **FFmpeg not found:**
   - Make sure FFmpeg is installed and added to your PATH
   - Restart your terminal/IDE after adding to PATH

2. **CUDA out of memory:**
   - Use CPU version: `export CUDA_VISIBLE_DEVICES=""`
   - Or reduce batch size in the script

3. **PyAnnote authentication error:**
   - Set up Hugging Face token as described in Step 6
   - Or use the fallback diarization (works without authentication)

4. **Memory issues with large videos:**
   - Process videos one at a time
   - Ensure sufficient disk space for output files

### Performance Tips

- **GPU acceleration:** Use CUDA-enabled PyTorch for 5-10x speedup
- **Video compression:** Use H.264 codec for smaller output files
- **Batch processing:** Process multiple videos sequentially to avoid memory issues

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 2GB free disk space per hour of video
- CPU (any modern processor)

**Recommended:**
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB+ free disk space
- SSD for faster I/O

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your video files are in supported formats
4. Check that you have sufficient disk space

For additional help, please refer to the project documentation or create an issue in the repository.
