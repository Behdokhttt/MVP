import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Dict, List, Tuple, Optional
import opensmile
import pandas as pd
from scipy import stats


class AudioFeatureExtractor:
    """
    Modular audio feature extraction pipeline for emotion classification.
    Combines traditional signal processing features with deep learning embeddings.
    """
    
    def __init__(self, sample_rate: int = 16000, use_transformer: bool = True):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            use_transformer: Whether to load transformer models for deep embeddings
        """
        self.sample_rate = sample_rate
        self.use_transformer = use_transformer
        
        # Initialize OpenSMILE for traditional features
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=4
        )
        
        # Initialize transformer models for deep embeddings
        if self.use_transformer:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model.eval()
    
    def load_and_preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        # Load audio with librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features and their statistics.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary containing MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Extract delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Compute statistics over time
        features = {}
        
        # Static MFCCs
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        features['mfcc_min'] = np.min(mfccs, axis=1)
        features['mfcc_max'] = np.max(mfccs, axis=1)
        features['mfcc_skew'] = stats.skew(mfccs, axis=1)
        features['mfcc_kurt'] = stats.kurtosis(mfccs, axis=1)
        
        # Delta MFCCs
        features['delta_mfcc_mean'] = np.mean(delta_mfccs, axis=1)
        features['delta_mfcc_std'] = np.std(delta_mfccs, axis=1)
        
        # Delta-delta MFCCs
        features['delta2_mfcc_mean'] = np.mean(delta2_mfccs, axis=1)
        features['delta2_mfcc_std'] = np.std(delta2_mfccs, axis=1)
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features for emotion classification.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary containing spectral features
        """
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral flux
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary containing energy features
        """
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_min'] = np.min(rms)
        features['rms_max'] = np.max(rms)
        
        return features
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features using librosa's piptrack.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary containing pitch features
        """
        features = {}
        
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        
        # Get the dominant pitch for each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch)
        
        pitch_values = np.array(pitch_values)
        
        # Remove unvoiced frames (pitch = 0)
        voiced_pitches = pitch_values[pitch_values > 0]
        
        if len(voiced_pitches) > 0:
            features['pitch_mean'] = np.mean(voiced_pitches)
            features['pitch_std'] = np.std(voiced_pitches)
            features['pitch_min'] = np.min(voiced_pitches)
            features['pitch_max'] = np.max(voiced_pitches)
            features['voicing_ratio'] = len(voiced_pitches) / len(pitch_values)
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
            features['voicing_ratio'] = 0.0
        
        return features
    
    def extract_opensmile_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract eGeMAPS features using OpenSMILE.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing OpenSMILE features
        """
        features_df = self.smile.process_file(audio_path)
        return features_df.iloc[0].to_dict()
    
    def extract_wav2vec_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Wav2Vec2 embeddings for deep learning features.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Wav2Vec2 embeddings
        """
        if not self.use_transformer:
            return np.array([])
        
        # Prepare input for Wav2Vec2
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            embeddings = self.wav2vec_model(**inputs).last_hidden_state
        
        # Pool embeddings (mean pooling over time dimension)
        pooled_embeddings = torch.mean(embeddings, dim=1).squeeze().numpy()
        
        return pooled_embeddings
    
    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all audio features for emotion classification.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load and preprocess audio
        audio = self.load_and_preprocess_audio(audio_path)
        
        # Extract traditional features
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        energy_features = self.extract_energy_features(audio)
        pitch_features = self.extract_pitch_features(audio)
        opensmile_features = self.extract_opensmile_features(audio_path)
        
        # Extract deep learning embeddings
        wav2vec_embeddings = self.extract_wav2vec_embeddings(audio)
        
        # Combine all features
        all_features = {}
        all_features.update(mfcc_features)
        all_features.update(spectral_features)
        all_features.update(energy_features)
        all_features.update(pitch_features)
        all_features.update(opensmile_features)
        
        if self.use_transformer:
            # Add wav2vec embeddings with proper naming
            for i, emb in enumerate(wav2vec_embeddings):
                all_features[f'wav2vec_emb_{i}'] = emb
        
        return all_features
    
    def extract_features_batch(self, audio_paths: List[str]) -> List[Dict[str, np.ndarray]]:
        """
        Extract features for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for audio_path in audio_paths:
            try:
                features = self.extract_all_features(audio_path)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                features_list.append({})
        
        return features_list


def main():
    """
    Example usage of the AudioFeatureExtractor.
    """
    # Initialize the feature extractor
    extractor = AudioFeatureExtractor(sample_rate=16000, use_transformer=True)
    
    # Example audio path (replace with actual audio file)
    audio_path = "example_audio.wav"
    
    try:
        # Extract features
        features = extractor.extract_all_features(audio_path)
        
        # Print feature information
        print(f"Extracted {len(features)} features from {audio_path}")
        print("\nFeature types:")
        
        feature_types = {}
        for key, value in features.items():
            feature_type = type(value).__name__
            if feature_type not in feature_types:
                feature_types[feature_type] = []
            feature_types[feature_type].append(key)
        
        for ftype, keys in feature_types.items():
            print(f"  {ftype}: {len(keys)} features")
        
        # Print some example features
        print("\nExample feature values:")
        example_keys = ['mfcc_mean_0', 'spectral_centroid_mean', 'pitch_mean', 'voicing_ratio']
        for key in example_keys:
            if key in features:
                print(f"  {key}: {features[key]}")
        
    except FileNotFoundError:
        print(f"Audio file not found: {audio_path}")
        print("Please replace 'example_audio.wav' with a valid audio file path.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
