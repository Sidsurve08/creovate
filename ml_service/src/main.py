from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from listing_pipeline import run_listing_pipeline
from price_prediction import predict_price
from speech_to_text import transcribe_audio
from nlp_models import classify_text, detect_language, encode_text
from image_analysis import classify_image, remove_background, enhance_image
from vector_search import get_recommendations, search_similar
from api_v2 import router as v2_router
from model_registry import load_all_models

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="CreoVate ML Service",
    description="Machine learning service for creative content analysis and classification",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v2 API router with trained model support
app.include_router(v2_router)

# Load models at startup (will fall back to rule-based if not available)
@app.on_event("startup")
async def startup_event():
    load_all_models()

# Pydantic models
class ClassificationRequest(BaseModel):
    content_type: str
    content: str
    metadata: Optional[dict] = None

class ClassificationResponse(BaseModel):
    success: bool
    category: str
    confidence: float
    tags: List[str]
    metadata: dict

class TextEmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = "indicbert"

class TextEmbeddingResponse(BaseModel):
    success: bool
    embedding: List[float]
    dimension: int

class ListingPipelineRequest(BaseModel):
    text: Optional[str] = None
    transcript_source: Optional[str] = "typed"

class RecommendationRequest(BaseModel):
    user_id: str
    content_type: Optional[str] = None
    limit: Optional[int] = 10

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[dict]
    metadata: dict

class PricePredictionRequest(BaseModel):
    category: str
    materials: List[str]
    techniques: List[str]
    time_required: Optional[int] = 20
    region: Optional[str] = "India"

class PricePredictionResponse(BaseModel):
    success: bool
    price_analysis: dict
    metadata: dict

@app.post("/ai-assist/price/suggest", response_model=PricePredictionResponse)
async def price_suggestion(request: PricePredictionRequest):
    """
    Suggest fair price using XGBoost model
    """
    try:
        price_analysis = predict_price(
            request.category,
            request.materials,
            request.techniques,
            request.time_required,
            request.region
        )
        
        return PricePredictionResponse(
            success=True,
            price_analysis=price_analysis,
            metadata={
                "processingTime": "50ms",
                "model": "XGBoost",
                "features": price_analysis['factors']
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CreoVate ML Service",
        "version": "1.0.0",
        "pipeline": "faster-whisper → IndicBERT → XGBoost → CLIP → rembg",
        "models": {
            "price_prediction": "XGBoost",
            "speech_to_text": "faster-whisper",
            "nlp": "IndicBERT + DistilBERT",
            "embeddings": "IndicBERT",
            "image_classification": "CLIP",
            "background_removal": "rembg",
            "image_enhancement": "PIL + OpenCV"
        },
        "features": {
            "regional_languages": ["hi", "mr", "ta", "te", "bn", "gu", "en"],
            "supported_categories": ["Pottery", "Textiles", "Painting", "Jewelry", "Sculpture", "Metalwork", "Woodwork", "Paper", "Leather", "Basketry", "Traditional Art"],
            "ml_algorithms": ["XGBoost", "IndicBERT", "DistilBERT", "CLIP", "faster-whisper", "rembg"]
        }
    }


@app.post("/pipeline/listing")
async def listing_pipeline(request: ListingPipelineRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    result = run_listing_pipeline(request.text.strip(), request.transcript_source or "typed")
    return {"success": True, "data": result}


@app.post("/speech/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Transcribe audio using faster-whisper for regional language support."""
    try:
        audio_data = await file.read()
        result = transcribe_audio(audio_data)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image classification endpoint
@app.post("/classify/image", response_model=ClassificationResponse)
async def classify_image_endpoint(file: UploadFile = File(...)):
    """
    Classify uploaded image using CLIP model
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        
        # Classify using CLIP
        categories = ['painting', 'sculpture', 'pottery', 'textile', 'jewelry',
                     'woodwork', 'metalwork', 'basketry', 'paper art', 'leather craft',
                     'traditional art', 'abstract art', 'realistic art', 'folk art']
        
        result = classify_image(image_data, categories)
        
        # Generate tags based on classification
        tags = [result['category']]
        if result['confidence'] > 0.7:
            tags.append('high-confidence')
        
        return ClassificationResponse(
            success=True,
            category=result['category'],
            confidence=result['confidence'],
            tags=tags,
            metadata={
                "model": result['engine'],
                "file_name": file.filename,
                "file_size": len(image_data),
                "all_scores": result.get('all_scores', {}),
                "image_size": result.get('image_size')
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text classification endpoint
@app.post("/classify/text", response_model=ClassificationResponse)
async def classify_text_endpoint(request: ClassificationRequest):
    """
    Classify text content using IndicBERT/DistilBERT
    """
    try:
        categories = ['Pottery', 'Textiles', 'Painting', 'Jewelry', 'Sculpture', 
                     'Metalwork', 'Woodwork', 'Paper', 'Leather', 'Basketry', 'Traditional Art']
        
        result = classify_text(request.content, categories)
        
        # Generate tags based on classification
        tags = [result['classification']]
        if result['confidence'] > 0.7:
            tags.append('high-confidence')
        
        return ClassificationResponse(
            success=True,
            category=result['classification'],
            confidence=result['confidence'],
            tags=tags,
            metadata={
                "model": result['engine'],
                "text_length": len(request.content),
                "all_scores": result.get('all_scores', {})
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text embedding endpoint
@app.post("/embeddings/text", response_model=TextEmbeddingResponse)
async def create_text_embedding(request: TextEmbeddingRequest):
    """
    Create text embeddings using IndicBERT/DistilBERT
    """
    try:
        embedding = encode_text(request.text, request.model)
        
        return TextEmbeddingResponse(
            success=True,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Image enhancement endpoint
@app.post("/image/enhance")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    """
    Enhance image quality using PIL and computer vision techniques
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        result = enhance_image(image_data)
        
        if result['success']:
            from fastapi.responses import Response
            return Response(
                content=result['enhanced_image'],
                media_type="image/jpeg",
                headers={
                    "X-Enhancement-Engine": result['engine'],
                    "X-Enhancements": ",".join(result.get('enhancements', []))
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Enhancement failed'))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background removal endpoint
@app.post("/image/remove-background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    """
    Remove background from image using rembg
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        result = remove_background(image_data)
        
        if result['success']:
            from fastapi.responses import Response
            return Response(
                content=result['processed_image'],
                media_type="image/png",
                headers={
                    "X-Background-Removal-Engine": result['engine'],
                    "X-Original-Size": f"{result['original_size'][0]}x{result['original_size'][1]}"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Background removal failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations using collaborative filtering
    """
    try:
        # TODO: Implement actual recommendation system
        mock_recommendations = [
            {
                "id": "1",
                "title": "Abstract Composition",
                "type": "artwork",
                "score": 0.95,
                "reason": "Based on your interest in abstract art"
            },
            {
                "id": "2",
                "title": "Digital Art Workshop",
                "type": "event",
                "score": 0.87,
                "reason": "Popular among creators with similar skills"
            }
        ]
        
        return RecommendationResponse(
            success=True,
            recommendations=mock_recommendations[:request.limit],
            metadata={
                "user_id": request.user_id,
                "algorithm": "collaborative-filtering",
                "total_recommendations": len(mock_recommendations)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Audio classification endpoint
@app.post("/classify/audio", response_model=ClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    """
    Classify audio content using YAMNet
    """
    try:
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be audio")
        
        # TODO: Implement actual YAMNet model inference
        return ClassificationResponse(
            success=True,
            category="ambient-music",
            confidence=0.85,
            tags=["ambient", "peaceful", "background"],
            metadata={
                "model": "YAMNet",
                "file_name": file.filename,
                "duration": "30s"  # mock duration
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal classification endpoint
@app.post("/classify/multimodal", response_model=ClassificationResponse)
async def classify_multimodal(files: List[UploadFile] = File(...)):
    """
    Classify multimodal content (image + text + audio)
    """
    try:
        # TODO: Implement actual multimodal classification
        return ClassificationResponse(
            success=True,
            category="multimedia-art-project",
            confidence=0.91,
            tags=["multimedia", "interactive", "digital"],
            metadata={
                "model": "CLIP + DistilBERT + YAMNet",
                "file_count": len(files),
                "content_types": [f.content_type for f in files]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )