"""
Base TTS engine interface and abstract classes.

Provides abstraction layer for different TTS engines (Piper, Coqui, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class TTSEngineType(Enum):
    """Types of TTS engines."""
    PIPER = "piper"
    COQUI = "coqui"
    ESPEAK = "espeak"
    EDGE_TTS = "edge_tts"
    FALLBACK = "fallback"


@dataclass
class TTSConfig:
    """Base configuration for TTS engines."""
    # Voice settings
    voice_id: str = "default"
    language: str = "en"
    gender: str = "neutral"  # male, female, neutral
    
    # Audio settings
    sample_rate: int = 22050
    output_format: str = "wav"  # wav, mp3, ogg
    
    # Generation settings
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    
    # Model settings
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Asynchronous settings
    use_async: bool = True
    max_queue_size: int = 10
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'voice_id': self.voice_id,
            'language': self.language,
            'gender': self.gender,
            'sample_rate': self.sample_rate,
            'output_format': self.output_format,
            'speed': self.speed,
            'pitch': self.pitch,
            'volume': self.volume,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'use_async': self.use_async,
            'max_queue_size': self.max_queue_size,
            'enable_cache': self.enable_cache,
            'cache_dir': self.cache_dir
        }


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio_data: bytes
    duration: float
    sample_rate: int
    format: str
    text: str
    voice_id: str
    language: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'format': self.format,
            'text': self.text,
            'voice_id': self.voice_id,
            'language': self.language,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata
        }


class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS engines.
    
    All TTS engine implementations must inherit from this class.
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize TTS engine.
        
        Args:
            config: TTS configuration
        """
        self.config = config
        self.is_initialized = False
        self.engine_type = TTSEngineType.FALLBACK
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the TTS engine.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Optional callback function for async results
            
        Returns:
            TTSResult with audio data
        """
        pass
    
    @abstractmethod
    def synthesize_async(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None
    ) -> None:
        """
        Synthesize speech asynchronously.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Callback function for results
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available voices for this engine.
        
        Returns:
            Dictionary of voice_id -> voice_info
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str) -> bool:
        """
        Set the active voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str) -> bool:
        """
        Set the language.
        
        Args:
            language: Language code (e.g., 'en', 'hi', 'bn')
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def is_available(self) -> bool:
        """
        Check if the engine is available and initialized.
        
        Returns:
            True if available, False otherwise
        """
        return self.is_initialized
    
    def get_engine_type(self) -> TTSEngineType:
        """Get the engine type."""
        return self.engine_type
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
