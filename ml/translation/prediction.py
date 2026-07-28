"""
Structured prediction objects for ISL translation.

Provides clean, structured objects for translation results independent from ML models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class PredictionStatus(Enum):
    """Status of a prediction."""
    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class TranslationResult:
    """
    Structured translation result from class ID to ISL word/phrase.
    
    Independent from ML model - only handles translation logic.
    """
    # Input
    class_id: int
    confidence: float
    probabilities: Optional[Dict[int, float]] = None
    
    # Translation
    translated_word: str = ""
    translated_phrase: str = ""
    language: str = "en"
    
    # Metadata
    status: PredictionStatus = PredictionStatus.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    inference_time_ms: Optional[float] = None
    
    # Additional context
    alternative_translations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, threshold: float = 0.5) -> bool:
        """
        Check if prediction is valid based on confidence threshold.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            True if valid, False otherwise
        """
        return self.confidence >= threshold and self.status == PredictionStatus.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all fields
        """
        return {
            'class_id': self.class_id,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'translated_word': self.translated_word,
            'translated_phrase': self.translated_phrase,
            'language': self.language,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'inference_time_ms': self.inference_time_ms,
            'alternative_translations': self.alternative_translations,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranslationResult':
        """
        Create TranslationResult from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TranslationResult instance
        """
        return cls(
            class_id=data['class_id'],
            confidence=data['confidence'],
            probabilities=data.get('probabilities'),
            translated_word=data.get('translated_word', ''),
            translated_phrase=data.get('translated_phrase', ''),
            language=data.get('language', 'en'),
            status=PredictionStatus(data.get('status', 'unknown')),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            inference_time_ms=data.get('inference_time_ms'),
            alternative_translations=data.get('alternative_translations', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class BatchTranslationResult:
    """
    Batch translation result for multiple predictions.
    """
    results: List[TranslationResult] = field(default_factory=list)
    batch_timestamp: datetime = field(default_factory=datetime.now)
    total_predictions: int = 0
    valid_predictions: int = 0
    average_confidence: float = 0.0
    
    def add_result(self, result: TranslationResult) -> None:
        """
        Add a translation result to the batch.
        
        Args:
            result: TranslationResult to add
        """
        self.results.append(result)
        self.total_predictions += 1
        if result.status == PredictionStatus.VALID:
            self.valid_predictions += 1
        self._update_average_confidence()
    
    def _update_average_confidence(self) -> None:
        """Update average confidence across all results."""
        if not self.results:
            self.average_confidence = 0.0
            return
        self.average_confidence = sum(r.confidence for r in self.results) / len(self.results)
    
    def get_valid_results(self, threshold: float = 0.5) -> List[TranslationResult]:
        """
        Get only valid results above threshold.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            List of valid TranslationResults
        """
        return [r for r in self.results if r.is_valid(threshold)]
    
    def get_most_frequent(self, threshold: float = 0.5) -> Optional[TranslationResult]:
        """
        Get the most frequent valid prediction.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            Most frequent TranslationResult or None
        """
        valid_results = self.get_valid_results(threshold)
        if not valid_results:
            return None
        
        # Count frequencies by class_id
        from collections import Counter
        class_counts = Counter(r.class_id for r in valid_results)
        most_common_class = class_counts.most_common(1)[0][0]
        
        # Return the first result with that class
        for r in valid_results:
            if r.class_id == most_common_class:
                return r
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all fields
        """
        return {
            'results': [r.to_dict() for r in self.results],
            'batch_timestamp': self.batch_timestamp.isoformat(),
            'total_predictions': self.total_predictions,
            'valid_predictions': self.valid_predictions,
            'average_confidence': self.average_confidence
        }
