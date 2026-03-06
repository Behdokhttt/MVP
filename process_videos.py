"""
Video Processing Script for Audio Feature Extraction
====================================================

This script processes video files from the input_videos folder,
extracts audio features, and saves them as pickle files in the output folder.
"""

import os
import pickle
import glob
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip
import tempfile
import shutil

from audio_features import AudioFeatureExtractor


class VideoProcessor:
    """
    Processes video files to extract audio features.
    """
    
    def __init__(self, input_dir: str = "data/input_videos", output_dir: str = "data/output/audio"):
        """
        Initialize the video processor.
        
        Args:
            input_dir: Directory containing input video files
            output_dir: Directory to save audio features
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.audio_extractor = AudioFeatureExtractor(sample_rate=16000, use_transformer=True)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Extract audio using moviepy
            with VideoFileClip(video_path) as clip:
                audio = clip.audio
                if audio is not None:
                    audio.write_audiofile(temp_audio.name, fps=16000, codec='pcm_s16le')
                    audio.close()
                else:
                    print(f"Warning: No audio found in {video_path}")
                    os.unlink(temp_audio.name)
                    return None
            
            return temp_audio.name
            
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
            return None
    
    def process_single_video(self, video_path: str) -> bool:
        """
        Process a single video file and extract audio features.
        
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
            
            # Extract audio features
            features = self.audio_extractor.extract_all_features(temp_audio_path)
            
            # Clean up temporary audio file
            os.unlink(temp_audio_path)
            
            # Generate output filename
            video_name = Path(video_path).stem
            output_path = self.output_dir / f"{video_name}_audio_features.pkl"
            
            # Save features as pickle file
            with open(output_path, 'wb') as f:
                pickle.dump(features, f)
            
            print(f"Saved features to: {output_path}")
            print(f"Extracted {len(features)} features")
            
            return True
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return False
    
    def process_all_videos(self) -> None:
        """
        Process all video files in the input directory.
        """
        print(f"Starting video processing...")
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
        print(f"Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir}")
    
    def get_processed_files(self) -> list:
        """
        Get list of already processed files.
        
        Returns:
            List of processed pickle files
        """
        return list(self.output_dir.glob("*_audio_features.pkl"))
    
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
    Main function to process all videos.
    """
    # Initialize processor
    processor = VideoProcessor()
    
    # Show current status
    video_files = processor.get_video_files()
    processed_files = processor.get_processed_files()
    
    print("Video Audio Feature Extraction")
    print("=" * 40)
    print(f"Video files found: {len(video_files)}")
    print(f"Already processed: {len(processed_files)}")
    print()
    
    # Process all videos
    processor.process_all_videos()


if __name__ == "__main__":
    main()
