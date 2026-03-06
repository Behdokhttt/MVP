"""
Fine-tuned Emotion Text Classifier
================================

This script implements a fine-tuned DeBERTa-v3 model for emotion classification from text.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from typing import Dict, List, Any, Tuple
from pathlib import Path
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionTextDataset(Dataset):
    """
    Custom dataset for emotion text classification.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmotionClassifier(nn.Module):
    """
    Fine-tunable DeBERTa-v3 model for emotion classification.
    """
    
    def __init__(self, model_name: str = 'microsoft/deberta-v3-base', num_classes: int = 7, dropout: float = 0.1):
        super(EmotionClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get DeBERTa outputs
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }


class FineTunedEmotionExtractor:
    """
    Extracts features using fine-tuned DeBERTa-v3 model for emotion classification.
    """
    
    def __init__(self, model_path: str = None, num_classes: int = 7):
        """
        Initialize the fine-tuned emotion extractor.
        
        Args:
            model_path: Path to fine-tuned model (if None, uses pre-trained)
            num_classes: Number of emotion classes
        """
        self.num_classes = num_classes
        self.emotion_labels = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from: {model_path}")
            self.model = EmotionClassifier(model_path, num_classes)
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Using pre-trained DeBERTa-v3 (not fine-tuned)")
            self.model = EmotionClassifier('microsoft/deberta-v3-base', num_classes)
        
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text using fine-tuned model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted features
        """
        if not text.strip():
            return self._get_empty_features()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get logits and probabilities
            logits = outputs['logits'].cpu().numpy().flatten()
            probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
            
            # Get hidden states for embeddings
            hidden_states = outputs['hidden_states'].cpu().numpy()
            
            # Create feature dictionary
            features = {
                'text_length': len(text),
                'has_text': len(text.strip()) > 0,
                'predicted_emotion_idx': np.argmax(logits),
                'predicted_emotion': self.emotion_labels[np.argmax(logits)],
                'confidence': float(np.max(probabilities)),
            }
            
            # Add emotion probabilities
            for i, emotion in enumerate(self.emotion_labels):
                features[f'prob_{emotion}'] = float(probabilities[i])
                features[f'logit_{emotion}'] = float(logits[i])
            
            # Add embeddings (mean pooling of hidden states)
            mean_embedding = np.mean(hidden_states, axis=1).flatten()
            cls_embedding = hidden_states[:, 0, :].flatten()  # [CLS] token
            
            for i, emb in enumerate(mean_embedding):
                features[f'embedding_mean_{i}'] = float(emb)
            
            for i, emb in enumerate(cls_embedding):
                features[f'embedding_cls_{i}'] = float(emb)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty features for invalid input."""
        features = {
            'text_length': 0,
            'has_text': False,
            'predicted_emotion_idx': -1,
            'predicted_emotion': 'unknown',
            'confidence': 0.0,
        }
        
        # Add emotion probabilities (all zero)
        for emotion in self.emotion_labels:
            features[f'prob_{emotion}'] = 0.0
            features[f'logit_{emotion}'] = 0.0
        
        # Add zero embeddings (768 dimensions for DeBERTa-v3-base)
        for i in range(768):
            features[f'embedding_mean_{i}'] = 0.0
            features[f'embedding_cls_{i}'] = 0.0
        
        return features
    
    def save_finetuned_model(self, save_path: str, optimizer_state=None, epoch=None):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
            optimizer_state: Optimizer state dict
            epoch: Current epoch number
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'num_classes': self.num_classes,
            'emotion_labels': self.emotion_labels
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")


def create_dummy_training_data():
    """
    Create dummy training data for demonstration.
    In practice, you would use your labeled emotion dataset.
    """
    texts = [
        "I am so happy and excited today!",
        "I feel very sad and depressed",
        "This makes me angry and frustrated",
        "I'm scared and terrified",
        "This is disgusting and revolting",
        "Wow, I'm so surprised and shocked",
        "I feel neutral and calm",
        "What a wonderful day, I'm joyful",
        "I'm feeling down and melancholic",
        "This is infuriating and makes me mad"
    ]
    
    labels = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]  # Corresponding emotion indices
    
    return texts, labels


def fine_tune_model():
    """
    Fine-tune DeBERTa-v3 on emotion data (demonstration).
    """
    print("Fine-tuning DeBERTa-v3 for emotion classification...")
    
    # Create dummy data (replace with your actual dataset)
    texts, labels = create_dummy_training_data()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = EmotionClassifier('microsoft/deberta-v3-base', num_classes=7)
    
    # Create dataset
    dataset = EmotionTextDataset(texts, labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./emotion_model',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Fine-tune
    trainer.train()
    
    # Save model
    model_path = './emotion_model/fine_tuned_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'num_classes': 7,
        'emotion_labels': ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
    }, model_path)
    
    print(f"Fine-tuned model saved to: {model_path}")
    return model_path


def main():
    """
    Main function to demonstrate fine-tuned emotion extraction.
    """
    print("Fine-tuned Emotion Text Feature Extraction")
    print("=" * 50)
    
    # Option 1: Fine-tune a model
    # model_path = fine_tune_model()
    
    # Option 2: Use pre-trained model (recommended for now)
    model_path = None
    
    # Initialize extractor
    extractor = FineTunedEmotionExtractor(model_path)
    
    # Test with sample texts
    test_texts = [
        "I am so happy and excited today!",
        "I feel very sad and depressed",
        "This makes me angry and frustrated",
        "I'm scared and terrified",
        "This is disgusting and revolting",
        "Wow, I'm so surprised and shocked",
        "I feel neutral and calm",
        ""  # Empty text test
    ]
    
    print("\nTesting emotion extraction:")
    print("-" * 50)
    
    for text in test_texts:
        features = extractor.extract_features(text)
        print(f"Text: '{text}'")
        print(f"Predicted: {features['predicted_emotion']} (confidence: {features['confidence']:.3f})")
        print(f"Probabilities: ", end="")
        for emotion in extractor.emotion_labels:
            prob = features[f'prob_{emotion}']
            print(f"{emotion}: {prob:.3f} ", end="")
        print("\n" + "-" * 30)


if __name__ == "__main__":
    main()
