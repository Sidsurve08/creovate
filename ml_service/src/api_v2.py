"""
ML Service v2 API Router
New endpoints with trained model support and fallbacks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from model_registry import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["v2"])

# Request/Response Models
class TextClassificationRequest(BaseModel):
    text: str

class TextClassificationResponse(BaseModel):
    category: str
    confidence: float
    source: str  # "trained_model" or "rule_based_fallback"

class PriceSuggestionRequest(BaseModel):
    category: Optional[str] = None
    craft: Optional[str] = None
    region: Optional[str] = None
    technique: Optional[str] = None
    complexity: Optional[int] = 1
    weight_g: Optional[float] = 0

class PriceSuggestionResponse(BaseModel):
    min_price: float
    suggested_price: float
    max_price: float
    confidence: float
    source: str
    similar_items: List[Dict[str, Any]] = []

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    index: str = "faiss_listings"  # or "faiss_heritage"

class SemanticSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    source: str

class ModelStatusResponse(BaseModel):
    models: Dict[str, str]

# Endpoints

@router.post("/classify/text", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    """Classify text into category using trained model or fallback"""
    model = registry.get_model("text_classifier")
    
    if model:
        try:
            # Use trained model
            import torch
            
            inputs = model["tokenizer"](
                request.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = model["model"](**inputs)
                predictions = outputs.logits
                probs = torch.softmax(predictions, dim=-1)
                confidence, predicted_class = torch.max(probs, dim=-1)
            
            category = model["label_encoder"].classes_[predicted_class.item()]
            
            return TextClassificationResponse(
                category=category,
                confidence=float(confidence),
                source="trained_model"
            )
        except Exception as e:
            logger.error(f"Model inference failed: {e}, falling back to rule-based")
    
    # Rule-based fallback
    text_lower = request.text.lower()
    
    # Simple keyword-based classification
    category_keywords = {
        "pottery": ["pottery", "clay", "ceramic", "wheel", "kiln"],
        "textiles": ["textile", "fabric", "weaving", "cloth", "silk", "cotton"],
        "paintings": ["painting", "art", "canvas", "watercolor", "oil"],
        "bamboo_cane": ["bamboo", "cane", "basket", "wicker"],
        "wood": ["wood", "carving", "sculpture", "timber"],
        "metalwork": ["metal", "brass", "copper", "iron", "forging"],
        "jewellery": ["jewelry", "jewellery", "gold", "silver", "gem"],
        "leather": ["leather", "bag", "shoe", "wallet"],
        "stone_terracotta": ["stone", "terracotta", "sculpture"],
        "toys": ["toy", "doll", "game"],
        "embroidery": ["embroidery", "stitch", "needlework"],
        "paper_scroll": ["paper", "scroll", "calligraphy"],
        "other": []
    }
    
    best_category = "other"
    max_matches = 0
    
    for category, keywords in category_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_category = category
    
    confidence = min(0.5 + (max_matches * 0.1), 0.9)
    
    return TextClassificationResponse(
        category=best_category,
        confidence=confidence,
        source="rule_based_fallback"
    )

@router.post("/price/suggest", response_model=PriceSuggestionResponse)
async def suggest_price(request: PriceSuggestionRequest):
    """Suggest price range using trained model or fallback"""
    model = registry.get_model("price_model")
    
    if model:
        try:
            import pandas as pd
            import numpy as np
            
            # Prepare features
            feature_data = {
                "complexity": request.complexity or 1,
                "weight_g": request.weight_g or 0,
                "handmade": True,
                "is_rental": False
            }
            
            # Encode categorical features
            le = model["label_encoders"]
            for col in ["category", "craft", "region", "technique"]:
                value = getattr(request, col, None)
                if value and col in le:
                    if value in le[col].classes_:
                        feature_data[f"{col}_encoded"] = le[col].transform([value])[0]
                    else:
                        feature_data[f"{col}_encoded"] = -1
                else:
                    feature_data[f"{col}_encoded"] = -1
            
            # Create DataFrame with correct columns
            df = pd.DataFrame([feature_data])
            for col in model["feature_cols"]:
                if col not in df.columns:
                    df[col] = 0
            df = df[model["feature_cols"]]
            
            # Predict
            suggested = model["model"].predict(df)[0]
            min_price = model["model_min"].predict(df)[0]
            max_price = model["model_max"].predict(df)[0]
            
            # Ensure reasonable values
            min_price = max(min_price, suggested * 0.7)
            max_price = max(max_price, suggested * 1.3)
            
            return PriceSuggestionResponse(
                min_price=float(min_price),
                suggested_price=float(suggested),
                max_price=float(max_price),
                confidence=0.75,
                source="trained_model"
            )
        except Exception as e:
            logger.error(f"Model inference failed: {e}, falling back to rule-based")
    
    # Rule-based fallback
    base_prices = {
        "pottery": 500,
        "textiles": 2000,
        "paintings": 3000,
        "bamboo_cane": 800,
        "wood": 1500,
        "metalwork": 2500,
        "jewellery": 5000,
        "leather": 1200,
        "stone_terracotta": 1000,
        "toys": 300,
        "embroidery": 1800,
        "paper_scroll": 600,
        "other": 1000
    }
    
    category = request.category or "other"
    base_price = base_prices.get(category, 1000)
    complexity = request.complexity or 1
    
    suggested = base_price * complexity
    min_price = suggested * 0.7
    max_price = suggested * 1.3
    
    return PriceSuggestionResponse(
        min_price=float(min_price),
        suggested_price=float(suggested),
        max_price=float(max_price),
        confidence=0.5,
        source="rule_based_fallback"
    )

@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """Semantic search using FAISS or fallback"""
    model = registry.get_model(f"faiss_{request.index}")
    
    if model:
        try:
            import numpy as np
            
            # Generate query embedding
            query_embedding = model["model"].encode([request.query])
            
            # Normalize for cosine similarity
            import faiss
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = request.top_k
            D, I = model["index"].search(query_embedding, k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(model["ids"]):
                    results.append({
                        "id": model["ids"][idx],
                        "score": float(score),
                        "rank": i + 1
                    })
            
            return SemanticSearchResponse(
                results=results,
                source="trained_model"
            )
        except Exception as e:
            logger.error(f"FAISS search failed: {e}, falling back to rule-based")
    
    # Rule-based fallback (simple keyword matching)
    # In a real implementation, this would search the database
    return SemanticSearchResponse(
        results=[],
        source="rule_based_fallback"
    )

@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get status of all loaded models"""
    return ModelStatusResponse(models=registry.get_status())
