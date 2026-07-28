"""
API routes for SilentVoice backend.

Defines all REST API endpoints for health check, prediction, and model information.
"""

import logging
import time
import uuid
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TranslationRequest,
    TranslationResponse,
    ErrorResponse,
    FileUploadResponse,
    MetricsResponse,
    TTSRequest,
    TTSResponse
)
from api.services import inference_service
from api.config import settings
from api.websocket import connection_manager, handle_websocket_prediction
from api.auth import get_current_api_key, require_full_access
from api.analytics import prediction_history

# Conditional TTS import
try:
    from api.tts import TTSManager, TTSConfig
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTSManager = None
    TTSConfig = None


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["API"])

# Initialize TTS manager
tts_manager = None
if settings.enable_tts and TTS_AVAILABLE:
    tts_manager = TTSManager(TTSConfig(language=settings.tts_language))

# Metrics tracking
metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_inference_time_ms": 0.0,
    "start_time": time.time()
}


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model loading state.
    
    **Example Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00",
        "version": "1.0.0",
        "model_loaded": true
    }
    ```
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=inference_service.is_model_loaded()
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model information.
    
    Returns details about the loaded model including architecture,
    configuration, and available classes.
    """
    result = inference_service.get_model_info()
    
    if not result['success']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.get('error', 'Model information unavailable')
        )
    
    return ModelInfoResponse(**result)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = get_current_api_key):
    """
    Make a single prediction.
    
    Accepts feature vectors and returns the predicted class with confidence score.
    
    **Example Request:**
    ```json
    {
        "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "return_probabilities": true,
        "language": "en"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "success": true,
        "predicted_class": 0,
        "predicted_label": "Hello",
        "confidence": 0.95,
        "probabilities": {"0": 0.95, "1": 0.05},
        "inference_time_ms": 15.2
    }
    ```
    """
    metrics["total_requests"] += 1
    
    result = inference_service.predict(
        features=request.features,
        return_probabilities=request.return_probabilities,
        language=request.language
    )
    
    if result['success']:
        metrics["successful_predictions"] += 1
        metrics["total_inference_time_ms"] += result['inference_time_ms']
        
        # Add to prediction history
        prediction_history.add_prediction(
            predicted_class=result['predicted_class'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            inference_time_ms=result['inference_time_ms'],
            language=request.language or settings.default_language,
            request_id=str(uuid.uuid4())
        )
    else:
        metrics["failed_predictions"] += 1
    
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, api_key: str = get_current_api_key):
    """
    Make batch predictions.
    
    Accepts multiple feature sequences and returns predictions for each.
    """
    metrics["total_requests"] += 1
    
    result = inference_service.predict_batch(
        sequences=request.sequences,
        return_probabilities=request.return_probabilities,
        language=request.language
    )
    
    if result['success']:
        metrics["successful_predictions"] += result['successful_predictions']
        metrics["failed_predictions"] += result['total_predictions'] - result['successful_predictions']
        metrics["total_inference_time_ms"] += result['total_inference_time_ms']
    else:
        metrics["failed_predictions"] += len(request.sequences)
    
    return BatchPredictionResponse(**result)


@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, api_key: str = get_current_api_key):
    """
    Translate class ID to ISL word/phrase.
    
    Accepts a class ID and returns the corresponding ISL translation.
    
    **Example Request:**
    ```json
    {
        "class_id": 0,
        "confidence": 0.85,
        "language": "hi"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "success": true,
        "class_id": 0,
        "translated_word": "Hello",
        "translated_phrase": "Namaste",
        "language": "hi",
        "confidence": 0.85,
        "status": "valid"
    }
    ```
    """
    result = inference_service.translate(
        class_id=request.class_id,
        confidence=request.confidence,
        language=request.language
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get('error', 'Translation failed')
        )
    
    return TranslationResponse(**result)


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    auto_predict: bool = True,
    api_key: str = require_full_access
):
    """
    Upload a file containing feature sequences.
    
    Accepts .npy files with feature sequences and optionally auto-predicts.
    
    **Example:**
    Upload a .npy file containing feature vectors.
    The API will automatically detect the format and make predictions if auto_predict is enabled.
    
    **Response:**
    ```json
    {
        "success": true,
        "filename": "features.npy",
        "file_id": "550e8400-e29b-41d4-a716-446655440000",
        "file_size": 12345,
        "content_type": "application/octet-stream",
        "prediction": {
            "success": true,
            "predicted_class": 0,
            "predicted_label": "Hello",
            "confidence": 0.92
        }
    }
    ```
    """
    import numpy as np
    
    file_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Save file
        upload_dir = Path("api/uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Try to load as numpy
        try:
            features = np.load(file_path)
            
            prediction = None
            if auto_predict and inference_service.is_model_loaded():
                # Convert to list format
                if features.ndim == 2:
                    features_list = features.tolist()
                else:
                    features_list = features.tolist()
                
                pred_result = inference_service.predict(features_list)
                prediction = PredictionResponse(**pred_result)
        
        except Exception as e:
            logger.warning(f"Could not load file as numpy: {e}")
            prediction = None
        
        return FileUploadResponse(
            success=True,
            filename=file.filename,
            file_id=file_id,
            file_size=file_size,
            content_type=file.content_type,
            prediction=prediction
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return FileUploadResponse(
            success=False,
            filename=file.filename,
            file_id=file_id,
            file_size=0,
            content_type="",
            error=str(e)
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(api_key: str = get_current_api_key):
    """
    Get API metrics and statistics.
    
    Returns request counts, prediction statistics, and server uptime.
    
    **Example Response:**
    ```json
    {
        "success": true,
        "total_requests": 1500,
        "successful_predictions": 1450,
        "failed_predictions": 50,
        "average_inference_time_ms": 12.5,
        "active_connections": 3,
        "uptime_seconds": 3600.5
    }
    ```
    """
    uptime = time.time() - metrics["start_time"]
    avg_inference = (
        metrics["total_inference_time_ms"] / metrics["successful_predictions"]
        if metrics["successful_predictions"] > 0
        else 0.0
    )
    
    return MetricsResponse(
        success=True,
        total_requests=metrics["total_requests"],
        successful_predictions=metrics["successful_predictions"],
        failed_predictions=metrics["failed_predictions"],
        average_inference_time_ms=avg_inference,
        active_connections=connection_manager.get_connection_count(),
        uptime_seconds=uptime
    )


@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, api_key: str = get_current_api_key):
    """
    Convert text to speech using TTS engine.
    
    Accepts text and returns audio data for playback.
    
    **Example Request:**
    ```json
    {
        "text": "Hello, how are you?",
        "language": "en",
        "voice_id": "default",
        "speed": 1.0
    }
    ```
    
    **Example Response:**
    ```json
    {
        "success": true,
        "text": "Hello, how are you?",
        "audio_duration": 2.5,
        "sample_rate": 22050,
        "format": "wav",
        "voice_id": "default",
        "language": "en",
        "generation_time_ms": 150.5
    }
    ```
    """
    if tts_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS not enabled"
        )
    
    try:
        result = tts_manager.synthesize(
            text=request.text,
            language=request.language
        )
        
        return TTSResponse(
            success=result.success,
            text=request.text,
            audio_duration=result.duration,
            sample_rate=result.sample_rate,
            format=result.format,
            voice_id=result.voice_id,
            language=result.language,
            generation_time_ms=result.inference_time_ms if hasattr(result, 'inference_time_ms') else 0.0,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time predictions.
    
    Supports streaming predictions and bidirectional communication.
    
    **Connection:**
    Connect to `ws://localhost:8000/api/v1/ws`
    
    **Message Format:**
    ```json
    {
        "type": "predict",
        "request_id": "unique-id",
        "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "return_probabilities": true,
        "language": "en"
    }
    ```
    
    **Supported Message Types:**
    - `predict`: Single prediction
    - `batch_predict`: Batch prediction
    - `translate`: Class ID translation
    - `ping`: Ping/pong for connection health
    
    **Response Format:**
    ```json
    {
        "type": "prediction_result",
        "request_id": "unique-id",
        "success": true,
        "predicted_class": 0,
        "predicted_label": "Hello",
        "confidence": 0.95
    }
    ```
    """
    client_id = str(uuid.uuid4())
    
    await connection_manager.connect(websocket, client_id)
    
    try:
        await handle_websocket_prediction(websocket, client_id)
    finally:
        connection_manager.disconnect(client_id)


@router.get("/analytics/history")
async def get_prediction_history(limit: int = 100, api_key: str = get_current_api_key):
    """
    Get recent prediction history.
    
    **Query Parameters:**
    - `limit`: Maximum number of predictions to return (default: 100)
    
    **Example Response:**
    ```json
    {
        "predictions": [
            {
                "timestamp": "2024-01-01T00:00:00",
                "predicted_class": 0,
                "predicted_label": "Hello",
                "confidence": 0.95,
                "inference_time_ms": 15.2,
                "language": "en"
            }
        ],
        "total_count": 100
    }
    ```
    """
    predictions = prediction_history.get_recent_predictions(limit=limit)
    
    return {
        "predictions": predictions,
        "total_count": len(predictions)
    }


@router.get("/analytics/statistics")
async def get_analytics_statistics(api_key: str = get_current_api_key):
    """
    Get prediction analytics statistics.
    
    **Example Response:**
    ```json
    {
        "total_predictions": 1500,
        "average_confidence": 0.87,
        "average_inference_time_ms": 12.5,
        "class_distribution": {
            "Hello": 450,
            "Thank you": 380,
            "Yes": 320,
            "No": 350
        },
        "unique_classes": 25
    }
    ```
    """
    return prediction_history.get_statistics()


@router.get("/analytics/class/{class_id}")
async def get_predictions_by_class(class_id: int, limit: int = 50, api_key: str = get_current_api_key):
    """
    Get predictions for a specific class.
    
    **Path Parameters:**
    - `class_id`: Class ID to filter by
    
    **Query Parameters:**
    - `limit`: Maximum number of predictions to return (default: 50)
    """
    predictions = prediction_history.get_predictions_by_class(class_id, limit=limit)
    
    return {
        "class_id": class_id,
        "predictions": predictions,
        "total_count": len(predictions)
    }


@router.get("/analytics/language/{language}")
async def get_predictions_by_language(language: str, limit: int = 50, api_key: str = get_current_api_key):
    """
    Get predictions for a specific language.
    
    **Path Parameters:**
    - `language`: Language code to filter by (e.g., 'en', 'hi')
    
    **Query Parameters:**
    - `limit`: Maximum number of predictions to return (default: 50)
    """
    predictions = prediction_history.get_predictions_by_language(language, limit=limit)
    
    return {
        "language": language,
        "predictions": predictions,
        "total_count": len(predictions)
    }


@router.get("/analytics/low-confidence")
async def get_low_confidence_predictions(threshold: float = 0.5, limit: int = 50, api_key: str = get_current_api_key):
    """
    Get predictions with low confidence.
    
    **Query Parameters:**
    - `threshold`: Confidence threshold (default: 0.5)
    - `limit`: Maximum number of predictions to return (default: 50)
    """
    predictions = prediction_history.get_low_confidence_predictions(threshold, limit)
    
    return {
        "threshold": threshold,
        "predictions": predictions,
        "total_count": len(predictions)
    }


@router.delete("/analytics/history")
async def clear_prediction_history(api_key: str = require_full_access):
    """
    Clear prediction history.
    
    Requires full access permissions.
    """
    prediction_history.clear_history()
    
    return {
        "success": True,
        "message": "Prediction history cleared"
    }


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info",
            "predict": "/api/v1/predict",
            "predict_batch": "/api/v1/predict/batch",
            "translate": "/api/v1/translate"
        }
    }
