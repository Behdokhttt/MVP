"""
Video Speech Transcription Script
=================================

This script processes video files from the input_videos folder,
extracts speech transcripts, and saves them as text files in the output folder.
"""

import os
import glob
from pathlib import Path
import tempfile
import shutil
import whisper
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import torch
import numpy as np
import librosa


class VideoTranscriber:
    """
    Transcribes speech from video files using Whisper and SpeechRecognition.
    """
    
    def __init__(self, input_dir: str = "data/input_videos", output_dir: str = "data/output/transcripts"):
        """
        Initialize the video transcriber.
        
        Args:
            input_dir: Directory containing input video files
            output_dir: Directory to save transcript files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Whisper model (using base model for good balance of speed/accuracy)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.whisper_model = whisper.load_model("base", device=self.device)
        
        # Initialize SpeechRecognition as backup
        self.recognizer = sr.Recognizer()
        
        # Supported video formats
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file and save as temporary WAV file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to temporary audio file
        """
        try:
            print(f"Extracting audio from video: {video_path}")
            print(f"Video file exists: {os.path.exists(video_path)}")
            if os.path.exists(video_path):
                print(f"Video file size: {os.path.getsize(video_path)} bytes")
            
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            print(f"Created temporary audio file: {temp_audio.name}")
            
            # Extract audio using moviepy
            print("Loading video with moviepy...")
            with VideoFileClip(video_path) as clip:
                print(f"Video duration: {clip.duration} seconds")
                audio = clip.audio
                if audio is not None:
                    print(f"Audio duration: {audio.duration} seconds")
                    print("Writing audio to WAV file...")
                    audio.write_audiofile(temp_audio.name, fps=16000, codec='pcm_s16le', verbose=False, logger=None)
                    audio.close()
                    
                    # Check if audio file was created successfully
                    if os.path.exists(temp_audio.name):
                        file_size = os.path.getsize(temp_audio.name)
                        print(f"Audio file created successfully: {file_size} bytes")
                        return temp_audio.name
                    else:
                        print("Audio file was not created")
                        return None
                else:
                    print(f"Warning: No audio found in {video_path}")
                    os.unlink(temp_audio.name)
                    return None
            
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def transcribe_with_whisper(self, audio_path: str) -> dict:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            print(f"Attempting Whisper transcription...")
            print(f"Audio file: {audio_path}")
            print(f"File exists: {os.path.exists(audio_path)}")
            print(f"File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
            
            # Check if FFmpeg is available
            try:
                import subprocess
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                print("FFmpeg is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("FFmpeg not found, using librosa-based audio loading")
                # Load audio with librosa and convert to numpy array
                audio_data, sr = librosa.load(audio_path, sr=16000)
                print(f"Audio loaded with librosa: {len(audio_data)} samples, {sr} Hz")
                
                # Transcribe using numpy array directly
                result = self.whisper_model.transcribe(audio_data, language="en", task="transcribe")
            else:
                # Use normal file-based transcription
                result = self.whisper_model.transcribe(
                    audio_path,
                    language="en",  # Assuming English, can be None for auto-detection
                    task="transcribe"
                )
            
            print(f"Whisper transcription successful")
            print(f"Detected language: {result.get('language', 'unknown')}")
            print(f"Text length: {len(result.get('text', ''))} characters")
            
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "method": "whisper"
            }
            
        except Exception as e:
            print(f"Whisper transcription failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def transcribe_with_speechrecognition(self, audio_path: str) -> dict:
        """
        Transcribe audio using SpeechRecognition as backup.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            print(f"Attempting SpeechRecognition transcription...")
            print(f"Audio file: {audio_path}")
            print(f"File exists: {os.path.exists(audio_path)}")
            
            with sr.AudioFile(audio_path) as source:
                print(f"Audio duration: {source.DURATION} seconds")
                audio_data = self.recognizer.record(source)
                print(f"Audio data recorded: {len(audio_data.get_raw_data())} bytes")
                
            # Try Google Speech Recognition first
            try:
                print("Attempting Google Speech Recognition...")
                text = self.recognizer.recognize_google(audio_data)
                print(f"Google Speech Recognition successful")
                print(f"Text length: {len(text)} characters")
                return {
                    "text": text,
                    "language": "en",
                    "segments": [],
                    "method": "google_speech"
                }
            except sr.RequestError as e:
                print(f"Google Speech Recognition request failed: {str(e)}")
                # Fallback to Sphinx (offline)
                try:
                    print("Attempting Sphinx recognition...")
                    text = self.recognizer.recognize_sphinx(audio_data)
                    print(f"Sphinx recognition successful")
                    print(f"Text length: {len(text)} characters")
                    return {
                        "text": text,
                        "language": "en",
                        "segments": [],
                        "method": "sphinx"
                    }
                except Exception as sphinx_e:
                    print(f"Sphinx recognition failed: {str(sphinx_e)}")
                    raise sphinx_e
            except sr.UnknownValueError as e:
                print(f"Google Speech Recognition could not understand audio: {str(e)}")
                raise e
                
        except Exception as e:
            print(f"SpeechRecognition transcription failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribe audio using available methods.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        print(f"Starting transcription for: {audio_path}")
        
        # Try Whisper first (more accurate)
        print("Trying Whisper transcription...")
        result = self.transcribe_with_whisper(audio_path)
        if result and result["text"].strip():
            print("Whisper transcription succeeded")
            return result
        else:
            print("Whisper transcription failed or returned empty text")
        
        # Fallback to SpeechRecognition
        print("Trying SpeechRecognition transcription...")
        result = self.transcribe_with_speechrecognition(audio_path)
        if result and result["text"].strip():
            print("SpeechRecognition transcription succeeded")
            return result
        else:
            print("SpeechRecognition transcription failed or returned empty text")
        
        # If all methods fail
        print("All transcription methods failed")
        return {
            "text": "",
            "language": "unknown",
            "segments": [],
            "method": "none",
            "error": "All transcription methods failed"
        }
    
    def process_single_video(self, video_path: str) -> bool:
        """
        Process a single video file and extract transcript.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing: {video_path}")
            
            # Extract audio from video
            temp_audio_path = self.extract_audio_from_video(video_path)
            if temp_audio_path is None:
                return False
            
            # Transcribe audio
            transcription_result = self.transcribe_audio(temp_audio_path)
            
            # Clean up temporary audio file
            os.unlink(temp_audio_path)
            
            # Generate output filename
            video_name = Path(video_path).stem
            output_path = self.output_dir / f"{video_name}_transcript.txt"
            
            # Save transcript as text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Video: {Path(video_path).name}\n")
                f.write(f"Transcription Method: {transcription_result['method']}\n")
                f.write(f"Language: {transcription_result['language']}\n")
                f.write("=" * 50 + "\n\n")
                
                if transcription_result.get("error"):
                    f.write(f"Error: {transcription_result['error']}\n")
                else:
                    f.write("TRANSCRIPT:\n")
                    f.write(transcription_result["text"])
                    
                    # Add segment information if available (from Whisper)
                    if transcription_result["segments"]:
                        f.write("\n\n" + "=" * 50 + "\n")
                        f.write("SEGMENTS:\n")
                        for i, segment in enumerate(transcription_result["segments"], 1):
                            f.write(f"\nSegment {i}:\n")
                            f.write(f"  Start: {segment.get('start', 0):.2f}s\n")
                            f.write(f"  End: {segment.get('end', 0):.2f}s\n")
                            f.write(f"  Text: {segment.get('text', '')}\n")
            
            print(f"Saved transcript to: {output_path}")
            print(f"Method: {transcription_result['method']}")
            print(f"Text length: {len(transcription_result['text'])} characters")
            
            return True
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return False
    
    def process_all_videos(self) -> None:
        """
        Process all video files in the input directory.
        """
        print(f"Starting video transcription...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all video files
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(glob.glob(str(self.input_dir / f"*{ext}")))
        
        if not video_files:
            print("No video files found in input directory!")
            return
        
        print(f"Found {len(video_files)} video files")
        print("-" * 50)
        
        # Process each video
        successful = 0
        failed = 0
        
        for video_path in video_files:
            if self.process_single_video(video_path):
                successful += 1
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Transcription complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir}")
    
    def get_processed_files(self) -> list:
        """
        Get list of already processed files.
        
        Returns:
            List of processed transcript files
        """
        return list(self.output_dir.glob("*_transcript.txt"))
    
    def get_video_files(self) -> list:
        """
        Get list of video files in input directory.
        
        Returns:
            List of video file paths
        """
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(glob.glob(str(self.input_dir / f"*{ext}")))
        return video_files


def main():
    """
    Main function to transcribe all videos.
    """
    # Initialize transcriber
    transcriber = VideoTranscriber()
    
    # Show current status
    video_files = transcriber.get_video_files()
    processed_files = transcriber.get_processed_files()
    
    print("Video Speech Transcription")
    print("=" * 40)
    print(f"Video files found: {len(video_files)}")
    print(f"Already processed: {len(processed_files)}")
    print()
    
    # Process all videos
    transcriber.process_all_videos()


if __name__ == "__main__":
    main()
