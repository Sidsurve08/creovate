"""
Model Registry for ML Service
Manages loading and availability of trained models with fallbacks
"""

from pathlib import Path
import json
import pickle
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

class ModelRegistry:
    """Registry for ML models with fallback support"""
    
    def __init__(self):
        self.models = {}
        self.model_status = {}
    
    def load_text_classifier(self) -> Dict[str, Any]:
        """Load text classifier if available"""
        model_path = MODELS_DIR / "category_distilbert"
        
        if not model_path.exists():
            logger.warning("Text classifier not found, will use rule-based fallback")
            self.model_status["text_classifier"] = "rule_based_fallback"
            return None
        
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            
            model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
            tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
            
            with open(model_path / "label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)
            
            self.models["text_classifier"] = {
                "model": model,
                "tokenizer": tokenizer,
                "label_encoder": label_encoder,
                "config": config
            }
            self.model_status["text_classifier"] = "trained_model"
            logger.info("Text classifier loaded successfully")
            return self.models["text_classifier"]
            
        except Exception as e:
            logger.error(f"Failed to load text classifier: {e}")
            self.model_status["text_classifier"] = "rule_based_fallback"
            return None
    
    def load_price_model(self) -> Dict[str, Any]:
        """Load price prediction model if available"""
        model_path = MODELS_DIR / "price_xgb"
        
        if not model_path.exists():
            logger.warning("Price model not found, will use rule-based fallback")
            self.model_status["price_model"] = "rule_based_fallback"
            return None
        
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor()
            model.load_model(str(model_path / "price_model.json"))
            
            model_min = xgb.XGBRegressor()
            model_min.load_model(str(model_path / "price_model_min.json"))
            
            model_max = xgb.XGBRegressor()
            model_max.load_model(str(model_path / "price_model_max.json"))
            
            with open(model_path / "label_encoders.pkl", "rb") as f:
                label_encoders = pickle.load(f)
            
            with open(model_path / "feature_cols.json", "r") as f:
                feature_cols = json.load(f)
            
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)
            
            self.models["price_model"] = {
                "model": model,
                "model_min": model_min,
                "model_max": model_max,
                "label_encoders": label_encoders,
                "feature_cols": feature_cols,
                "config": config
            }
            self.model_status["price_model"] = "trained_model"
            logger.info("Price model loaded successfully")
            return self.models["price_model"]
            
        except Exception as e:
            logger.error(f"Failed to load price model: {e}")
            self.model_status["price_model"] = "rule_based_fallback"
            return None
    
    def load_faiss_index(self, index_name: str) -> Dict[str, Any]:
        """Load FAISS index if available"""
        import faiss
        from sentence_transformers import SentenceTransformer
        
        index_path = MODELS_DIR / f"{index_name}.index"
        model_path = MODELS_DIR / f"{index_name}_model"
        ids_path = MODELS_DIR / f"{index_name}_ids.pkl"
        
        if not index_path.exists() or not model_path.exists():
            logger.warning(f"FAISS index {index_name} not found, will use rule-based fallback")
            self.model_status[f"faiss_{index_name}"] = "rule_based_fallback"
            return None
        
        try:
            index = faiss.read_index(str(index_path))
            model = SentenceTransformer(str(model_path))
            
            with open(ids_path, "rb") as f:
                ids = pickle.load(f)
            
            metadata_path = MODELS_DIR / f"{index_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            self.models[f"faiss_{index_name}"] = {
                "index": index,
                "model": model,
                "ids": ids,
                "metadata": metadata
            }
            self.model_status[f"faiss_{index_name}"] = "trained_model"
            logger.info(f"FAISS index {index_name} loaded successfully")
            return self.models[f"faiss_{index_name}"]
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index {index_name}: {e}")
            self.model_status[f"faiss_{index_name}"] = "rule_based_fallback"
            return None
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all models"""
        return self.model_status
    
    def get_model(self, model_name: str):
        """Get a loaded model"""
        return self.models.get(model_name)

# Global registry instance
registry = ModelRegistry()

def load_all_models():
    """Load all available models at startup"""
    logger.info("Loading ML models...")
    registry.load_text_classifier()
    registry.load_price_model()
    registry.load_faiss_index("faiss_listings")
    registry.load_faiss_index("faiss_heritage")
    logger.info(f"Model loading complete. Status: {registry.get_status()}")
