"""
Piper TTS engine implementation.

Fast, offline neural TTS using Piper models.
"""

import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

from ml.tts.base import BaseTTSEngine, TTSConfig, TTSResult, TTSEngineType


logger = logging.getLogger(__name__)


class PiperTTSEngine(BaseTTSEngine):
    """
    Piper TTS engine implementation.
    
    Fast, offline neural TTS with support for multiple languages and voices.
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize Piper TTS engine.
        
        Args:
            config: TTS configuration
        """
        super().__init__(config)
        self.engine_type = TTSEngineType.PIPER
        
        self.piper_model = None
        self.synthesize_func = None
        
        # Async queue
        self.task_queue = queue.Queue(maxsize=config.max_queue_size)
        self.worker_thread = None
        self.is_running = False
        
        # Voice cache
        self.available_voices = {}
        self.current_voice = None
        
        logger.info("Initialized Piper TTS engine")
    
    def initialize(self) -> bool:
        """
        Initialize Piper TTS engine.
        
        Returns:
            True if successful, False otherwise
        """
        if not PIPER_AVAILABLE:
            logger.error("Piper is not available. Install with: pip install piper-tts")
            return False
        
        try:
            # Load model if path provided
            if self.config.model_path:
                self._load_model(self.config.model_path)
            else:
                logger.warning("No model path provided. Use set_voice() to load a model.")
            
            # Start async worker if enabled
            if self.config.use_async:
                self._start_async_worker()
            
            # Discover available voices
            self._discover_voices()
            
            self.is_initialized = True
            logger.info("Piper TTS engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            return False
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load Piper model from path.
        
        Args:
            model_path: Path to Piper model (.onnx file)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load Piper model
            # Note: This is a simplified interface - actual Piper API may differ
            self.piper_model = piper.PiperModel(str(model_path))
            self.current_voice = str(model_path)
            
            logger.info(f"Loaded Piper model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Piper model: {e}")
            return False
    
    def _discover_voices(self) -> None:
        """Discover available Piper voices."""
        # Common Piper voice directories
        voice_dirs = [
            Path.home() / ".local" / "share" / "piper" / "voices",
            Path("/usr/share/piper/voices"),
            Path("./piper_voices"),
        ]
        
        for voice_dir in voice_dirs:
            if voice_dir.exists():
                for voice_file in voice_dir.glob("*.onnx"):
                    voice_id = voice_file.stem
                    self.available_voices[voice_id] = {
                        'path': str(voice_file),
                        'language': self._extract_language_from_name(voice_id),
                        'gender': self._extract_gender_from_name(voice_id)
                    }
        
        # Add language-specific voice mappings for common ISL languages
        self._add_language_specific_mappings()
        
        logger.info(f"Discovered {len(self.available_voices)} Piper voices")
    
    def _add_language_specific_mappings(self) -> None:
        """Add language-specific voice mappings for ISL languages."""
        # Common Piper voice names for different languages
        # These are examples - actual voices depend on what's installed
        language_mappings = {
            'en': ['en_US-lessac-medium', 'en_US-amy-medium', 'en_US-kathleen-low'],
            'hi': ['hi_IN-govind-medium', 'hi_IN-cmu-indic-medium'],
            'bn': ['bn_IN-bishal-medium', 'bn_IND-cmu-indic-medium'],
            'ta': ['ta_IN-karthik-medium', 'ta_IND-cmu-indic-medium'],
            'te': ['te_IN-anjali-medium', 'te_IND-cmu-indic-medium'],
            'mr': ['mr_IND-cmu-indic-medium'],
            'gu': ['gu_IND-cmu-indic-medium'],
        }
        
        # Store mappings for language-based voice selection
        self.language_voice_mappings = language_mappings
    
    def _extract_language_from_name(self, voice_name: str) -> str:
        """Extract language code from voice name."""
        # Common patterns: en_US, hi_IN, etc.
        parts = voice_name.split('_')
        if len(parts) >= 1:
            return parts[0]
        return "en"
    
    def _extract_gender_from_name(self, voice_name: str) -> str:
        """Extract gender from voice name."""
        if 'male' in voice_name.lower() or 'man' in voice_name.lower():
            return 'male'
        elif 'female' in voice_name.lower() or 'woman' in voice_name.lower():
            return 'female'
        return 'neutral'
    
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
            callback: Optional callback function
            
        Returns:
            TTSResult with audio data
        """
        if not self.is_initialized or self.piper_model is None:
            return TTSResult(
                audio_data=b"",
                duration=0.0,
                sample_rate=self.config.sample_rate,
                format=self.config.output_format,
                text=text,
                voice_id=self.config.voice_id,
                language=self.config.language,
                success=False,
                error="Engine not initialized or no model loaded"
            )
        
        try:
            # Synthesize with Piper
            # Note: This is a simplified interface - actual Piper API may differ
            audio_data = self.piper_model.synthesize(
                text,
                speaker_id=self.config.voice_id,
                length_scale=1.0 / self.config.speed
            )
            
            # Convert to bytes
            audio_bytes = self._audio_to_bytes(audio_data)
            
            # Calculate duration
            duration = len(audio_data) / self.config.sample_rate
            
            # Save to file if path provided
            if output_path:
                self._save_audio(audio_bytes, output_path)
            
            result = TTSResult(
                audio_data=audio_bytes,
                duration=duration,
                sample_rate=self.config.sample_rate,
                format=self.config.output_format,
                text=text,
                voice_id=self.config.voice_id,
                language=self.config.language,
                success=True
            )
            
            if callback:
                callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return TTSResult(
                audio_data=b"",
                duration=0.0,
                sample_rate=self.config.sample_rate,
                format=self.config.output_format,
                text=text,
                voice_id=self.config.voice_id,
                language=self.config.language,
                success=False,
                error=str(e)
            )
    
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
        if not self.config.use_async:
            # Fallback to synchronous
            result = self.synthesize(text, output_path, callback)
            return
        
        task = {
            'text': text,
            'output_path': output_path,
            'callback': callback
        }
        
        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            logger.warning("Task queue is full, dropping request")
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
                    error="Task queue is full"
                ))
    
    def _start_async_worker(self) -> None:
        """Start async worker thread."""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Started async worker thread")
    
    def _worker_loop(self) -> None:
        """Worker loop for async synthesis."""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break
                
                text = task['text']
                output_path = task['output_path']
                callback = task['callback']
                
                result = self.synthesize(text, output_path, None)
                
                if callback:
                    callback(result)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def _audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert audio array to bytes.
        
        Args:
            audio_data: Audio array
            
        Returns:
            Audio bytes
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def _save_audio(self, audio_bytes: bytes, output_path: Path) -> None:
        """
        Save audio to file.
        
        Args:
            audio_bytes: Audio bytes
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == 'wav':
            import wave
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_bytes)
        else:
            # For other formats, would need additional libraries
            logger.warning(f"Format {self.config.output_format} not directly supported, saving as raw")
            output_path.write_bytes(audio_bytes)
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available voices.
        
        Returns:
            Dictionary of voice_id -> voice_info
        """
        return self.available_voices
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set the active voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if successful, False otherwise
        """
        if voice_id in self.available_voices:
            voice_info = self.available_voices[voice_id]
            model_path = voice_info['path']
            
            if self._load_model(model_path):
                self.config.voice_id = voice_id
                logger.info(f"Set voice to {voice_id}")
                return True
        
        # Try loading as direct path
        if Path(voice_id).exists():
            if self._load_model(voice_id):
                self.config.voice_id = voice_id
                return True
        
        logger.error(f"Voice not found: {voice_id}")
        return False
    
    def set_language(self, language: str) -> bool:
        """
        Set the language.
        
        Args:
            language: Language code
            
        Returns:
            True if successful, False otherwise
        """
        self.config.language = language
        
        # First try language-specific mappings
        if hasattr(self, 'language_voice_mappings') and language in self.language_voice_mappings:
            for voice_id in self.language_voice_mappings[language]:
                if voice_id in self.available_voices:
                    if self.set_voice(voice_id):
                        logger.info(f"Set language to {language} using voice {voice_id}")
                        return True
        
        # Fallback to searching available voices by language code
        for voice_id, voice_info in self.available_voices.items():
            if voice_info.get('language') == language:
                if self.set_voice(voice_id):
                    return True
        
        logger.warning(f"No voice found for language: {language}")
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False
        
        if self.worker_thread:
            # Signal worker to stop
            self.task_queue.put(None)
            self.worker_thread.join(timeout=5.0)
        
        if self.piper_model:
            del self.piper_model
            self.piper_model = None
        
        logger.info("Piper TTS engine cleaned up")
