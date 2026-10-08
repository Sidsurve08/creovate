"""NLP models using IndicBERT and DistilBERT for text processing.

This module implements actual transformer models for text classification,
language detection, and feature extraction.
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline
)
from typing import Dict, List, Any, Optional
import numpy as np

class NLPModelManager:
    """Manager for NLP models including IndicBERT and DistilBERT."""
    
    def __init__(self):
        """Initialize NLP models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Models
        self.indicbert_tokenizer = None
        self.indicbert_model = None
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self.classification_pipeline = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load NLP models with fallback support."""
        try:
            # Try to load IndicBERT for Indian languages
            print("Loading IndicBERT model...")
            self.indicbert_tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/indic-bert", 
                use_fast=True
            )
            self.indicbert_model = AutoModel.from_pretrained(
                "ai4bharat/indic-bert"
            ).to(self.device)
            print("IndicBERT loaded successfully")
        except Exception as e:
            print(f"Error loading IndicBERT: {e}")
            print("Falling back to multilingual BERT")
            try:
                self.indicbert_tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-multilingual-cased",
                    use_fast=True
                )
                self.indicbert_model = AutoModel.from_pretrained(
                    "bert-base-multilingual-cased"
                ).to(self.device)
                print("Multilingual BERT loaded as fallback")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.indicbert_tokenizer = None
                self.indicbert_model = None
        
        try:
            # Load DistilBERT for efficient classification
            print("Loading DistilBERT model...")
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased",
                use_fast=True
            )
            self.distilbert_model = AutoModel.from_pretrained(
                "distilbert-base-uncased"
            ).to(self.device)
            print("DistilBERT loaded successfully")
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
            self.distilbert_tokenizer = None
            self.distilbert_model = None
        
        try:
            # Load classification pipeline
            print("Loading classification pipeline...")
            self.classification_pipeline = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Classification pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading classification pipeline: {e}")
            self.classification_pipeline = None
    
    def encode_text(self, text: str, model_type: str = "indicbert") -> np.ndarray:
        """Encode text using transformer models.
        
        Args:
            text: Input text to encode
            model_type: Which model to use (indicbert, distilbert)
            
        Returns:
            Text embedding as numpy array
        """
        if model_type == "indicbert" and self.indicbert_tokenizer and self.indicbert_model:
            tokenizer = self.indicbert_tokenizer
            model = self.indicbert_model
        elif self.distilbert_tokenizer and self.distilbert_model:
            tokenizer = self.distilbert_tokenizer
            model = self.distilbert_model
        else:
            # Fallback to simple encoding
            return self._simple_encode(text)
        
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return self._simple_encode(text)
    
    def _simple_encode(self, text: str) -> np.ndarray:
        """Simple fallback encoding using character frequencies."""
        # Create a simple 128-dimensional embedding based on character frequencies
        embedding = np.zeros(128)
        for char in text[:512]:  # Limit to first 512 characters
            embedding[ord(char) % 128] += 1
        # Normalize
        if embedding.sum() > 0:
            embedding = embedding / embedding.sum()
        return embedding
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Classify text into categories.
        
        Args:
            text: Input text to classify
            categories: List of possible categories
            
        Returns:
            Classification result with scores
        """
        if self.classification_pipeline:
            try:
                # Use the pipeline for sentiment/text classification
                result = self.classification_pipeline(text)
                return {
                    'success': True,
                    'classification': result[0]['label'],
                    'confidence': result[0]['score'],
                    'engine': 'distilbert-pipeline'
                }
            except Exception as e:
                print(f"Error in pipeline classification: {e}")
        
        # Fallback to similarity-based classification
        return self._rule_based_classification(text, categories)
    
    def _rule_based_classification(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Rule-based classification as fallback."""
        text_lower = text.lower()
        category_scores = {}
        
        # Define keywords for each category
        category_keywords = {
            'Pottery': ['clay', 'pottery', 'ceramic', 'wheel', 'kiln', 'terracotta', 'mitti'],
            'Textiles': ['textile', 'fabric', 'weaving', 'loom', 'embroidery', 'saree', 'cotton'],
            'Painting': ['painting', 'canvas', 'brush', 'warli', 'madhubani', 'art'],
            'Jewelry': ['jewelry', 'necklace', 'earring', 'bangle', 'gemstone'],
            'Basketry': ['basket', 'bamboo', 'cane', 'weaving'],
            'Woodwork': ['wood', 'carving', 'furniture', 'teak'],
            'Traditional Art': ['warli', 'madhubani', 'kalamkari', 'tanjore', 'folk']
        }
        
        for category in categories:
            keywords = category_keywords.get(category, [])
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize score
            category_scores[category] = score / max(len(keywords), 1)
        
        # Find best category
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        confidence = min(0.9, max(0.5, best_score))
        
        return {
            'success': True,
            'classification': best_category,
            'confidence': confidence,
            'all_scores': category_scores,
            'engine': 'rule-based-fallback'
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Language detection result
        """
        # Unicode-based language detection
        unicode_ranges = {
            'hi': (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit)
            'bn': (0x0980, 0x09FF),  # Bengali
            'ta': (0x0B80, 0x0BFF),  # Tamil
            'te': (0x0C00, 0x0C7F),  # Telugu
            'mr': (0x0900, 0x097F),  # Marathi (same as Hindi)
            'gu': (0x0A80, 0x0AFF),  # Gujarati
            'pa': (0x0A00, 0x0A7F),  # Punjabi
        }
        
        char_counts = {}
        for char in text:
            code = ord(char)
            for lang, (start, end) in unicode_ranges.items():
                if start <= code <= end:
                    char_counts[lang] = char_counts.get(lang, 0) + 1
                    break
        
        if char_counts:
            detected_lang = max(char_counts, key=char_counts.get)
            confidence = char_counts[detected_lang] / len(text)
        else:
            detected_lang = 'en'
            confidence = 0.8
        
        # Special handling for Marathi vs Hindi
        if detected_lang == 'hi':
            marathi_indicators = ['मी', 'आहे', 'बांबू', 'टोकरी']
            if any(indicator in text for indicator in marathi_indicators):
                detected_lang = 'mr'
        
        return {
            'language': detected_lang,
            'confidence': confidence,
            'char_counts': char_counts,
            'engine': 'unicode-detection'
        }
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text using NLP models.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        language_result = self.detect_language(text)
        
        # Get embeddings
        embedding = self.encode_text(text)
        
        # Extract materials and techniques
        materials = self._extract_entities(text, 'materials')
        techniques = self._extract_entities(text, 'techniques')
        
        return {
            'language': language_result,
            'embedding': embedding.tolist(),
            'embedding_dim': len(embedding),
            'materials': materials,
            'techniques': techniques,
            'text_length': len(text),
            'word_count': len(text.split()),
            'engine': 'indicbert-distilbert'
        }
    
    def _extract_entities(self, text: str, entity_type: str) -> List[str]:
        """Extract entities from text (simplified implementation)."""
        entity_keywords = {
            'materials': {
                'Clay': ['clay', 'mitti', 'terracotta', 'ceramic'],
                'Fabric': ['fabric', 'cotton', 'silk', 'textile'],
                'Wood': ['wood', 'bamboo', 'teak'],
                'Metal': ['metal', 'brass', 'copper', 'bronze'],
                'Stone': ['stone', 'marble'],
                'Paint': ['paint', 'pigment', 'dye']
            },
            'techniques': {
                'Hand-painted': ['hand-painted', 'hand painted', 'paint'],
                'Hand-carved': ['hand-carved', 'hand carved', 'carving'],
                'Handwoven': ['handwoven', 'hand woven', 'weaving'],
                'Natural dye': ['natural dye', 'natural', 'dye'],
                'Kiln-fired': ['kiln', 'fired', 'baked']
            }
        }
        
        text_lower = text.lower()
        entities = []
        
        if entity_type in entity_keywords:
            for entity, keywords in entity_keywords[entity_type].items():
                if any(keyword in text_lower for keyword in keywords):
                    entities.append(entity)
        
        return entities

# Global NLP model manager
_nlp_manager = None

def get_nlp_manager() -> NLPModelManager:
    """Get or create the global NLP manager instance."""
    global _nlp_manager
    if _nlp_manager is None:
        _nlp_manager = NLPModelManager()
    return _nlp_manager

def encode_text(text: str, model_type: str = "indicbert") -> np.ndarray:
    """Convenience function for text encoding."""
    manager = get_nlp_manager()
    return manager.encode_text(text, model_type)

def classify_text(text: str, categories: List[str]) -> Dict[str, Any]:
    """Convenience function for text classification."""
    manager = get_nlp_manager()
    return manager.classify_text(text, categories)

def detect_language(text: str) -> Dict[str, Any]:
    """Convenience function for language detection."""
    manager = get_nlp_manager()
    return manager.detect_language(text)