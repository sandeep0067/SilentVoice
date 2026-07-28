"""
Fallback TTS engine for when primary engines are unavailable.

Provides a simple fallback implementation that can be extended or replaced.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from ml.tts.base import BaseTTSEngine, TTSConfig, TTSResult, TTSEngineType


logger = logging.getLogger(__name__)


class FallbackTTSEngine(BaseTTSEngine):
    """
    Fallback TTS engine for when primary engines are unavailable.
    
    This is a minimal implementation that can be used as a placeholder
    or extended with actual TTS functionality.
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize fallback TTS engine.
        
        Args:
            config: TTS configuration
        """
        super().__init__(config)
        self.engine_type = TTSEngineType.FALLBACK
        
        logger.info("Initialized Fallback TTS engine")
    
    def initialize(self) -> bool:
        """
        Initialize fallback TTS engine.
        
        Returns:
            True if successful
        """
        # Fallback engine is always "available" but doesn't produce actual audio
        self.is_initialized = True
        logger.warning("Using fallback TTS engine - no actual audio will be produced")
        return True
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None
    ) -> TTSResult:
        """
        Synthesize speech (fallback - no actual audio).
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Optional callback function
            
        Returns:
            TTSResult with empty audio data
        """
        logger.warning(f"Fallback synthesis called for: {text}")
        
        result = TTSResult(
            audio_data=b"",
            duration=0.0,
            sample_rate=self.config.sample_rate,
            format=self.config.output_format,
            text=text,
            voice_id=self.config.voice_id,
            language=self.config.language,
            success=False,
            error="Fallback engine - no actual TTS available"
        )
        
        if callback:
            callback(result)
        
        return result
    
    def synthesize_async(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None
    ) -> None:
        """
        Synthesize speech asynchronously (fallback).
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Callback function for results
        """
        # Fallback to synchronous
        result = self.synthesize(text, output_path, callback)
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available voices (fallback - empty).
        
        Returns:
            Empty dictionary
        """
        return {}
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set the active voice (fallback - always fails).
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            False
        """
        logger.warning(f"Cannot set voice in fallback engine: {voice_id}")
        return False
    
    def set_language(self, language: str) -> bool:
        """
        Set the language (fallback - always fails).
        
        Args:
            language: Language code
            
        Returns:
            False
        """
        logger.warning(f"Cannot set language in fallback engine: {language}")
        return False
