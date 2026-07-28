"""
Text-to-Speech module for ISL application.

Provides offline TTS with Piper and fallback support for other engines.
"""

from ml.tts.base import (
    BaseTTSEngine,
    TTSConfig,
    TTSResult,
    TTSEngineType
)
from ml.tts.piper_engine import PiperTTSEngine
from ml.tts.fallback_engine import FallbackTTSEngine
from ml.tts.manager import TTSManager

__all__ = [
    'BaseTTSEngine',
    'TTSConfig',
    'TTSResult',
    'TTSEngineType',
    'PiperTTSEngine',
    'FallbackTTSEngine',
    'TTSManager',
]
