"""
TTS manager for coordinating multiple TTS engines.

Provides a unified interface for TTS with automatic fallback and engine selection.
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

from ml.tts.base import BaseTTSEngine, TTSConfig, TTSResult, TTSEngineType
from ml.tts.piper_engine import PiperTTSEngine
from ml.tts.fallback_engine import FallbackTTSEngine


logger = logging.getLogger(__name__)


class TTSManager:
    """
    Manager for TTS engines with automatic fallback.
    
    Coordinates multiple TTS engines and provides a unified interface.
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize TTS manager.
        
        Args:
            config: TTS configuration
        """
        self.config = config or TTSConfig()
        
        # Available engines
        self.engines: Dict[TTSEngineType, BaseTTSEngine] = {}
        self.primary_engine: Optional[BaseTTSEngine] = None
        
        # Initialize engines
        self._initialize_engines()
        
        logger.info(f"TTS Manager initialized with {len(self.engines)} engines")
    
    def _initialize_engines(self) -> None:
        """Initialize available TTS engines."""
        # Try Piper first (preferred)
        piper_engine = PiperTTSEngine(self.config)
        if piper_engine.initialize():
            self.engines[TTSEngineType.PIPER] = piper_engine
            self.primary_engine = piper_engine
            logger.info("Piper TTS engine initialized successfully")
        else:
            logger.warning("Piper TTS engine not available")
        
        # Always add fallback
        fallback_engine = FallbackTTSEngine(self.config)
        fallback_engine.initialize()
        self.engines[TTSEngineType.FALLBACK] = fallback_engine
        
        # If no primary engine, use fallback
        if self.primary_engine is None:
            self.primary_engine = fallback_engine
            logger.warning("Using fallback TTS engine")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None,
        engine_type: Optional[TTSEngineType] = None
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Optional callback function
            engine_type: Specific engine to use (uses primary if None)
            
        Returns:
            TTSResult with audio data
        """
        # Select engine
        if engine_type is not None and engine_type in self.engines:
            engine = self.engines[engine_type]
        else:
            engine = self.primary_engine
        
        if engine is None:
            logger.error("No TTS engine available")
            return TTSResult(
                audio_data=b"",
                duration=0.0,
                sample_rate=self.config.sample_rate,
                format=self.config.output_format,
                text=text,
                voice_id=self.config.voice_id,
                language=self.config.language,
                success=False,
                error="No TTS engine available"
            )
        
        # Try synthesis with primary engine
        result = engine.synthesize(text, output_path, callback)
        
        # Fallback to fallback engine if primary failed
        if not result.success and engine.engine_type != TTSEngineType.FALLBACK:
            logger.warning(f"Primary engine failed, trying fallback")
            fallback = self.engines.get(TTSEngineType.FALLBACK)
            if fallback:
                result = fallback.synthesize(text, output_path, callback)
        
        return result
    
    def synthesize_async(
        self,
        text: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[TTSResult], None]] = None,
        engine_type: Optional[TTSEngineType] = None
    ) -> None:
        """
        Synthesize speech asynchronously.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            callback: Callback function for results
            engine_type: Specific engine to use (uses primary if None)
        """
        # Select engine
        if engine_type is not None and engine_type in self.engines:
            engine = self.engines[engine_type]
        else:
            engine = self.primary_engine
        
        if engine is None:
            logger.error("No TTS engine available")
            if callback:
                callback(TTSResult(
                    audio_data=b"",
                    duration=0.0,
                    sample_rate=self.config.sample_rate,
                    format=self.config.output_format,
                    text=text,
                    voice_id=self.config.voice_id,
                    language=self.config.language,
                    success=False,
                    error="No TTS engine available"
                ))
            return
        
        engine.synthesize_async(text, output_path, callback)
    
    def get_available_engines(self) -> List[TTSEngineType]:
        """
        Get list of available engine types.
        
        Returns:
            List of available engine types
        """
        return list(self.engines.keys())
    
    def get_primary_engine(self) -> Optional[BaseTTSEngine]:
        """Get the primary TTS engine."""
        return self.primary_engine
    
    def set_primary_engine(self, engine_type: TTSEngineType) -> bool:
        """
        Set the primary TTS engine.
        
        Args:
            engine_type: Engine type to set as primary
            
        Returns:
            True if successful, False otherwise
        """
        if engine_type in self.engines:
            self.primary_engine = self.engines[engine_type]
            logger.info(f"Set primary engine to {engine_type.value}")
            return True
        return False
    
    def get_available_voices(self, engine_type: Optional[TTSEngineType] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available voices from an engine.
        
        Args:
            engine_type: Specific engine (uses primary if None)
            
        Returns:
            Dictionary of voice_id -> voice_info
        """
        engine = self.engines.get(engine_type) if engine_type else self.primary_engine
        if engine:
            return engine.get_available_voices()
        return {}
    
    def set_voice(self, voice_id: str, engine_type: Optional[TTSEngineType] = None) -> bool:
        """
        Set the active voice.
        
        Args:
            voice_id: Voice identifier
            engine_type: Specific engine (uses primary if None)
            
        Returns:
            True if successful, False otherwise
        """
        engine = self.engines.get(engine_type) if engine_type else self.primary_engine
        if engine:
            return engine.set_voice(voice_id)
        return False
    
    def set_language(self, language: str, engine_type: Optional[TTSEngineType] = None) -> bool:
        """
        Set the language.
        
        Args:
            language: Language code
            engine_type: Specific engine (uses primary if None)
            
        Returns:
            True if successful, False otherwise
        """
        engine = self.engines.get(engine_type) if engine_type else self.primary_engine
        if engine:
            return engine.set_language(language)
        return False
    
    def update_config(self, **kwargs) -> None:
        """
        Update TTS configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Reinitialize engines if needed
        if 'model_path' in kwargs or 'voice_id' in kwargs:
            self._initialize_engines()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get TTS manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'available_engines': [e.value for e in self.get_available_engines()],
            'primary_engine': self.primary_engine.engine_type.value if self.primary_engine else None,
            'current_voice': self.config.voice_id,
            'current_language': self.config.language,
            'use_async': self.config.use_async,
            'sample_rate': self.config.sample_rate
        }
    
    def cleanup(self) -> None:
        """Clean up all engines."""
        for engine in self.engines.values():
            engine.cleanup()
        logger.info("TTS Manager cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
