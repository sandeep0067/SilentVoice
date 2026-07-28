"""
Pydantic schemas for request/response validation.

Defines the data models for API requests and responses.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


# Health check schemas
class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


# Model information schemas
class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    num_classes: int = Field(..., description="Number of output classes")
    feature_dim: int = Field(..., description="Input feature dimension")
    architecture: str = Field(..., description="Model architecture")
    device: str = Field(..., description="Device running on")
    checkpoint_path: str = Field(..., description="Model checkpoint path")
    class_names: Optional[List[str]] = Field(None, description="List of class names")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration")


# Prediction request schemas
class PredictionRequest(BaseModel):
    """Prediction request."""
    features: List[List[float]] = Field(
        ...,
        description="Feature vectors (sequence of frames)",
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return full probability distribution",
        example=False
    )
    language: Optional[str] = Field(
        default=None,
        description="Language for translation (uses default if None)",
        example="en"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature vectors."""
        if not v:
            raise ValueError("Features cannot be empty")
        if not all(isinstance(frame, list) for frame in v):
            raise ValueError("Features must be a list of lists")
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""
    success: bool = Field(..., description="Whether prediction was successful")
    predicted_class: int = Field(..., description="Predicted class ID")
    predicted_label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: Optional[Dict[int, float]] = Field(None, description="Full probability distribution")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


# Batch prediction schemas
class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    sequences: List[List[List[float]]] = Field(
        ...,
        description="List of feature sequences"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return probability distributions"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language for translation"
    )
    
    @validator('sequences')
    def validate_sequences(cls, v):
        """Validate feature sequences."""
        if not v:
            raise ValueError("Sequences cannot be empty")
        if not all(isinstance(seq, list) for seq in v):
            raise ValueError("Sequences must be a list of lists")
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    success: bool = Field(..., description="Whether batch prediction was successful")
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    total_inference_time_ms: float = Field(..., description="Total inference time")
    average_inference_time_ms: float = Field(..., description="Average inference time per prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


# Translation schemas
class TranslationRequest(BaseModel):
    """Translation request."""
    class_id: int = Field(
        ...,
        description="Class ID to translate",
        example=0
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score",
        example=0.85
    )
    language: Optional[str] = Field(
        default=None,
        description="Target language (uses default if None)",
        example="hi"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class TranslationResponse(BaseModel):
    """Translation response."""
    success: bool = Field(..., description="Whether translation was successful")
    class_id: int = Field(..., description="Input class ID")
    translated_word: str = Field(..., description="Translated word")
    translated_phrase: str = Field(..., description="Translated phrase")
    language: str = Field(..., description="Translation language")
    confidence: float = Field(..., description="Input confidence")
    status: str = Field(..., description="Prediction status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Translation timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


# Error response schemas
class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Validation error schemas
class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    success: bool = Field(default=False, description="Always False for validation errors")
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(default="Request validation failed", description="Error message")
    details: List[ValidationErrorDetail] = Field(..., description="Validation error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# File upload schemas
class FileUploadResponse(BaseModel):
    """File upload response."""
    success: bool = Field(..., description="Whether upload was successful")
    filename: str = Field(..., description="Uploaded filename")
    file_id: str = Field(..., description="Unique file identifier")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    prediction: Optional[PredictionResponse] = Field(None, description="Prediction result if auto-predicted")
    timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


# Metrics schemas
class MetricsResponse(BaseModel):
    """Metrics response."""
    success: bool = Field(..., description="Whether metrics retrieval was successful")
    total_requests: int = Field(..., description="Total number of requests")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    average_inference_time_ms: float = Field(..., description="Average inference time")
    active_connections: int = Field(..., description="Number of active WebSocket connections")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")


# TTS schemas
class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(..., description="Text to synthesize")
    language: str = Field(default="en", description="Target language")
    voice_id: Optional[str] = Field(None, description="Voice identifier")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text length."""
        if len(v) > 500:
            raise ValueError("Text must be less than 500 characters")
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        return v


class TTSResponse(BaseModel):
    """Text-to-speech response."""
    success: bool = Field(..., description="Whether TTS was successful")
    text: str = Field(..., description="Input text")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate")
    format: str = Field(..., description="Audio format")
    voice_id: str = Field(..., description="Voice used")
    language: str = Field(..., description="Language")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="TTS timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
