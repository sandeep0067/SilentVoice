from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    version: str


class InferenceRequest(BaseModel):
    frame_data: str
    timestamp: float


class InferenceResponse(BaseModel):
    gesture: str
    confidence: float
    timestamp: float


class TranslationRequest(BaseModel):
    gesture_sequence: list[str]
    context: Optional[str] = None


class TranslationResponse(BaseModel):
    text: str
    confidence: float


class TTSRequest(BaseModel):
    text: str
    voice_profile: Optional[str] = "default"


class TTSResponse(BaseModel):
    audio_data: str
    format: str
