"""
Prediction history and analytics tracking.

Stores recent predictions and provides analytics endpoints.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction record."""
    timestamp: datetime
    predicted_class: int
    predicted_label: str
    confidence: float
    inference_time_ms: float
    language: str
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PredictionHistory:
    """
    Manages prediction history and analytics.
    
    Stores recent predictions in memory and provides analytics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize prediction history.
        
        Args:
            max_history: Maximum number of predictions to store
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        
        # Statistics
        self.class_counts: Dict[int, int] = {}
        self.total_predictions = 0
        self.total_inference_time_ms = 0.0
        
        logger.info(f"PredictionHistory initialized with max_history={max_history}")
    
    def add_prediction(
        self,
        predicted_class: int,
        predicted_label: str,
        confidence: float,
        inference_time_ms: float,
        language: str,
        request_id: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> None:
        """
        Add a prediction to history.
        
        Args:
            predicted_class: Predicted class ID
            predicted_label: Predicted class label
            confidence: Prediction confidence
            inference_time_ms: Inference time in milliseconds
            language: Language used
            request_id: Request identifier
            client_ip: Client IP address
        """
        record = PredictionRecord(
            timestamp=datetime.now(),
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            language=language,
            request_id=request_id,
            client_ip=client_ip
        )
        
        self.history.append(record)
        
        # Update statistics
        self.class_counts[predicted_class] = self.class_counts.get(predicted_class, 0) + 1
        self.total_predictions += 1
        self.total_inference_time_ms += inference_time_ms
        
        logger.debug(f"Added prediction: class={predicted_class}, label={predicted_label}")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent predictions.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        recent = list(self.history)[-limit:]
        return [record.to_dict() for record in reversed(recent)]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of predicted classes.
        
        Returns:
            Dictionary of class_label -> count
        """
        # Convert class IDs to labels
        distribution = {}
        for class_id, count in self.class_counts.items():
            # Find a recent prediction with this class to get the label
            label = f"Class_{class_id}"
            for record in reversed(self.history):
                if record.predicted_class == class_id:
                    label = record.predicted_label
                    break
            distribution[label] = count
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get prediction statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.total_predictions == 0:
            return {
                "total_predictions": 0,
                "average_confidence": 0.0,
                "average_inference_time_ms": 0.0,
                "class_distribution": {},
                "unique_classes": 0
            }
        
        avg_confidence = sum(r.confidence for r in self.history) / len(self.history)
        avg_inference = self.total_inference_time_ms / self.total_predictions
        
        return {
            "total_predictions": self.total_predictions,
            "average_confidence": avg_confidence,
            "average_inference_time_ms": avg_inference,
            "class_distribution": self.get_class_distribution(),
            "unique_classes": len(self.class_counts),
            "history_size": len(self.history)
        }
    
    def get_predictions_by_class(self, class_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get predictions for a specific class.
        
        Args:
            class_id: Class ID to filter by
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        filtered = [r for r in self.history if r.predicted_class == class_id]
        recent = filtered[-limit:]
        return [record.to_dict() for record in reversed(recent)]
    
    def get_predictions_by_language(self, language: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get predictions for a specific language.
        
        Args:
            language: Language to filter by
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        filtered = [r for r in self.history if r.language == language]
        recent = filtered[-limit:]
        return [record.to_dict() for record in reversed(recent)]
    
    def get_low_confidence_predictions(self, threshold: float = 0.5, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get predictions with low confidence.
        
        Args:
            threshold: Confidence threshold
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        filtered = [r for r in self.history if r.confidence < threshold]
        recent = filtered[-limit:]
        return [record.to_dict() for record in reversed(recent)]
    
    def clear_history(self) -> None:
        """Clear prediction history."""
        self.history.clear()
        self.class_counts.clear()
        self.total_predictions = 0
        self.total_inference_time_ms = 0.0
        logger.info("Prediction history cleared")
    
    def get_history_size(self) -> int:
        """Get current history size."""
        return len(self.history)


# Global prediction history instance
prediction_history = PredictionHistory(max_history=1000)
