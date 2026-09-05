"""
Text-to-Speech Handler for SilentVoice ASL Translator.

This module provides thread-safe TTS functionality using pyttsx3
to avoid blocking the main inference loop during speech playback.
"""

import threading
import queue
import pyttsx3
from typing import Optional


class TTSHandler:
    """
    Thread-safe text-to-speech handler using pyttsx3.
    
    Runs TTS in a separate thread to avoid blocking the main video loop.
    """
    
    def __init__(self, clear_after_speaking: bool = False):
        """
        Initialize TTS handler.
        
        Args:
            clear_after_speaking: Whether to clear the word buffer after speaking
        """
        self.clear_after_speaking = clear_after_speaking
        self.speaking = False
        self.speech_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        
        # Initialize pyttsx3 engine
        try:
            self.engine = pyttsx3.init()
            # Configure voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available (often clearer for TTS)
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate (adjustable)
            self.engine.setProperty('rate', 150)  # Words per minute
            self.engine.setProperty('volume', 0.9)  # Volume level (0-1)
            
            print("TTS engine initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize TTS engine: {e}")
            self.engine = None
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text in a separate thread.
        
        Args:
            text: Text to speak
            
        Returns:
            True if speech was queued successfully, False otherwise
        """
        if not self.engine:
            print("TTS engine not available")
            return False
        
        if not text or not text.strip():
            print("Empty text, skipping speech")
            return False
        
        # Queue the speech request
        self.speech_queue.put(text)
        
        # Start speech thread if not already running
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.stop_event.clear()
            self.tts_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.tts_thread.start()
        
        return True
    
    def _speech_worker(self):
        """Worker thread that processes speech requests from the queue."""
        while not self.stop_event.is_set():
            try:
                # Get text from queue with timeout
                text = self.speech_queue.get(timeout=0.1)
                
                if text:
                    self.speaking = True
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"Error during speech: {e}")
                    finally:
                        self.speaking = False
                        self.speech_queue.task_done()
                
            except queue.Empty:
                # No speech in queue, continue loop
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
                continue
    
    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking."""
        return self.speaking
    
    def stop(self):
        """Stop any ongoing speech and cleanup."""
        self.stop_event.set()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def test_tts():
    """Test the TTS handler."""
    print("Testing TTS Handler")
    print("=" * 50)
    
    try:
        tts = TTSHandler(clear_after_speaking=False)
        
        if tts.engine:
            print("TTS engine available")
            print("Speaking test message...")
            tts.speak("Hello, this is a test of the text to speech system.")
            
            # Wait for speech to complete
            import time
            while tts.is_speaking():
                print("Speaking...")
                time.sleep(0.5)
            
            print("Speech completed")
            tts.stop()
        else:
            print("TTS engine not available - install pyttsx3 first")
            print("Run: pip install pyttsx3")
        
        print("=" * 50)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install pyttsx3 first: pip install pyttsx3")
        print("=" * 50)


if __name__ == '__main__':
    test_tts()
