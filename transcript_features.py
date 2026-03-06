"""
Speech Transcription Feature Extraction
===================================

This script processes transcript files from the transcripts folder,
extracts linguistic and emotional features, and saves them as pickle files.
"""

import os
import pickle
import glob
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import nltk
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch


class TranscriptFeatureExtractor:
    """
    Extracts features from speech transcripts for emotion classification.
    """
    
    def __init__(self, input_dir: str = "data/output/transcripts", output_dir: str = "data/output/transcript_features"):
        """
        Initialize the transcript feature extractor.
        
        Args:
            input_dir: Directory containing transcript files
            output_dir: Directory to save feature files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except OSError:
            print("Downloading spaCy model...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                print("spaCy model downloaded and loaded successfully")
            except (subprocess.CalledProcessError, Exception) as e:
                print(f"Failed to download spaCy model: {str(e)}")
                print("Please run: python -m spacy download en_core_web_sm")
                # Fallback to basic processing without spaCy
                self.nlp = None
        
        # Initialize transformer model for embeddings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
            self.bert_model = AutoModel.from_pretrained('microsoft/deberta-v3-base')
            self.bert_model.eval()
            print("DeBERTa-v3 model loaded successfully (using slow tokenizer)")
        except Exception as e:
            print(f"Failed to load DeBERTa-v3 with slow tokenizer: {str(e)}")
            # Fallback to BERT-base
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
                self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
                self.bert_model.eval()
                print("Fallback to BERT-base successful")
            except Exception as e2:
                print(f"Failed to load BERT-base: {str(e2)}")
                # Final fallback - no transformer model
                self.tokenizer = None
                self.bert_model = None
        
        # Initialize fine-tuned emotion classifier
        try:
            from emotion_text_classifier import FineTunedEmotionExtractor
            self.emotion_classifier = FineTunedEmotionExtractor()
            self.use_finetuned = True
            print("Using fine-tuned DeBERTa-v3 for emotion classification")
        except ImportError:
            self.use_finetuned = False
            print("Using frozen DeBERTa-v3 embeddings (fine-tuned model not available)")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Emotion-related word lists
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'glad', 'cheerful', 'delighted', 'pleased', 'satisfied'],
            'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'sorrowful', 'melancholy'],
            'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'enraged'],
            'fear': ['afraid', 'scared', 'fearful', 'terrified', 'anxious', 'worried', 'nervous'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned']
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def parse_transcript_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse transcript file and extract text content.
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Dictionary containing parsed transcript data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract transcript text (after the separators)
            lines = content.split('\n')
            transcript_text = ""
            method = "unknown"
            language = "unknown"
            
            for i, line in enumerate(lines):
                if line.startswith("Transcription Method:"):
                    method = line.split(":")[1].strip()
                elif line.startswith("Language:"):
                    language = line.split(":")[1].strip()
                elif line.startswith("TRANSCRIPT:"):
                    # Get everything after "TRANSCRIPT:"
                    transcript_text = '\n'.join(lines[i+1:]).strip()
                    break
            
            return {
                'text': transcript_text,
                'method': method,
                'language': language,
                'raw_content': content
            }
            
        except Exception as e:
            print(f"Error parsing transcript file {file_path}: {str(e)}")
            return None
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic linguistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        if not text.strip():
            return {}
        
        # Basic text statistics
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
            'punctuation_count': len(re.findall(r'[^\w\s]', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment features
        """
        if not text.strip():
            return {}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'sentiment_positive': max(0, polarity),
            'sentiment_negative': max(0, -polarity),
            'sentiment_neutral': 1 - abs(polarity)
        }
    
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """
        Extract part-of-speech features using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of POS features
        """
        if not text.strip() or self.nlp is None:
            return {}
        
        doc = self.nlp(text)
        
        # Count different POS tags
        pos_counts = {
            'nouns': len([token for token in doc if token.pos_ == 'NOUN']),
            'verbs': len([token for token in doc if token.pos_ == 'VERB']),
            'adjectives': len([token for token in doc if token.pos_ == 'ADJ']),
            'adverbs': len([token for token in doc if token.pos_ == 'ADV']),
            'pronouns': len([token for token in doc if token.pos_ == 'PRON']),
            'conjunctions': len([token for token in doc if token.pos_ == 'CCONJ']),
            'interjections': len([token for token in doc if token.pos_ == 'INTJ'])
        }
        
        total_tokens = len(doc)
        if total_tokens > 0:
            # Convert to ratios
            pos_ratios = {f'{k}_ratio': v/total_tokens for k, v in pos_counts.items()}
            pos_counts.update(pos_ratios)
        
        return pos_counts
    
    def extract_emotion_keywords(self, text: str) -> Dict[str, int]:
        """
        Count emotion-related keywords in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion keyword counts
        """
        if not text.strip():
            return {}
        
        text_lower = text.lower()
        emotion_counts = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_counts[f'{emotion}_keywords'] = count
        
        return emotion_counts
    
    def extract_bert_embeddings(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings for text or use fine-tuned emotion classifier.
        
        Args:
            text: Input text
            
        Returns:
            BERT embedding vector or emotion features
        """
        if not text.strip():
            if self.use_finetuned:
                return self.emotion_classifier._get_empty_features()
            else:
                return np.zeros(768)  # DeBERTa-v3 hidden size
        
        try:
            if self.use_finetuned:
                # Use fine-tuned emotion classifier
                emotion_features = self.emotion_classifier.extract_features(text)
                return emotion_features
            else:
                # Use frozen DeBERTa-v3 embeddings
                if self.tokenizer is not None and self.bert_model is not None:
                    # Tokenize text
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                        
                    # Use mean pooling of the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    
                    return embeddings
                else:
                    # No transformer model available - return zero embeddings
                    print("Warning: No transformer model available, using zero embeddings")
                    return np.zeros(768)
            
        except Exception as e:
            print(f"Error extracting embeddings: {str(e)}")
            if self.use_finetuned:
                return self.emotion_classifier._get_empty_features()
            else:
                return np.zeros(768)
    
    def extract_all_features(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from transcript data.
        
        Args:
            transcript_data: Parsed transcript data
            
        Returns:
            Dictionary containing all extracted features
        """
        text = transcript_data['text']
        
        # Extract all feature types
        linguistic_features = self.extract_linguistic_features(text)
        sentiment_features = self.extract_sentiment_features(text)
        pos_features = self.extract_pos_features(text)
        emotion_keywords = self.extract_emotion_keywords(text)
        bert_embeddings = self.extract_bert_embeddings(text)
        
        # Combine all features
        all_features = {
            'transcript_method': transcript_data['method'],
            'transcript_language': transcript_data['language'],
            'has_text': len(text.strip()) > 0
        }
        
        # Add numerical features
        all_features.update(linguistic_features)
        all_features.update(sentiment_features)
        all_features.update(pos_features)
        all_features.update(emotion_keywords)
        
        # Add BERT embeddings or emotion features
        if self.use_finetuned and isinstance(bert_embeddings, dict):
            # Fine-tuned model returns dictionary of emotion features
            all_features.update(bert_embeddings)
        else:
            # Frozen embeddings return numpy array
            for i, emb in enumerate(bert_embeddings):
                all_features[f'bert_emb_{i}'] = emb
        
        return all_features
    
    def process_single_transcript(self, transcript_path: str) -> bool:
        """
        Process a single transcript file and extract features.
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing: {transcript_path}")
            
            # Parse transcript file
            transcript_data = self.parse_transcript_file(transcript_path)
            if transcript_data is None:
                return False
            
            # Extract features
            features = self.extract_all_features(transcript_data)
            
            # Generate output filename
            transcript_name = Path(transcript_path).stem
            output_path = self.output_dir / f"{transcript_name}_features.pkl"
            
            # Save features as pickle file
            with open(output_path, 'wb') as f:
                pickle.dump(features, f)
            
            print(f"Saved features to: {output_path}")
            print(f"Extracted {len(features)} features")
            print(f"Text length: {len(transcript_data['text'])} characters")
            
            return True
            
        except Exception as e:
            print(f"Error processing {transcript_path}: {str(e)}")
            return False
    
    def process_all_transcripts(self) -> None:
        """
        Process all transcript files in the input directory.
        """
        print(f"Starting transcript feature extraction...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all transcript files
        transcript_files = list(self.input_dir.glob("*_transcript.txt"))
        
        if not transcript_files:
            print("No transcript files found in input directory!")
            return
        
        print(f"Found {len(transcript_files)} transcript files")
        print("-" * 50)
        
        # Process each transcript
        successful = 0
        failed = 0
        
        for transcript_path in transcript_files:
            if self.process_single_transcript(str(transcript_path)):
                successful += 1
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Feature extraction complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_dir}")
    
    def get_processed_files(self) -> list:
        """
        Get list of already processed files.
        
        Returns:
            List of processed feature files
        """
        return list(self.output_dir.glob("*_features.pkl"))
    
    def get_transcript_files(self) -> list:
        """
        Get list of transcript files in input directory.
        
        Returns:
            List of transcript file paths
        """
        return list(self.input_dir.glob("*_transcript.txt"))


def main():
    """
    Main function to extract features from all transcripts.
    """
    # Initialize feature extractor
    extractor = TranscriptFeatureExtractor()
    
    # Show current status
    transcript_files = extractor.get_transcript_files()
    processed_files = extractor.get_processed_files()
    
    print("Speech Transcription Feature Extraction")
    print("=" * 50)
    print(f"Transcript files found: {len(transcript_files)}")
    print(f"Already processed: {len(processed_files)}")
    print()
    
    # Process all transcripts
    extractor.process_all_transcripts()


if __name__ == "__main__":
    main()
