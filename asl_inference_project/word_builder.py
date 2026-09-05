"""
WordBuilder class for stability-based word construction from ASL predictions.

This class implements a stability-based approach to convert noisy frame-by-frame
predictions into clean words by requiring consecutive stable predictions.
"""

from typing import Tuple


class WordBuilder:
    """
    Builds sentences from ASL alphabet predictions using stability checking.
    
    Only accepts a prediction as final if the same letter is predicted for
    N consecutive frames with confidence above a threshold.
    """
    
    # Default configuration
    DEFAULT_STABILITY_FRAMES = 5
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, stability_frames: int = None, confidence_threshold: float = None):
        """
        Initialize WordBuilder.
        
        Args:
            stability_frames: Number of consecutive frames required to confirm a letter
            confidence_threshold: Minimum confidence required to consider a prediction
        """
        self.stability_frames = stability_frames or self.DEFAULT_STABILITY_FRAMES
        self.confidence_threshold = confidence_threshold or self.DEFAULT_CONFIDENCE_THRESHOLD
        
        # State tracking
        self.sentence_buffer = ""
        self.current_letter = None
        self.stability_counter = 0
        self.last_confirmed_letter = None
        
    def update(self, predicted_label: str, confidence: float) -> Tuple[str, bool]:
        """
        Update sentence builder with a new prediction.
        
        Args:
            predicted_label: Predicted class label (e.g., 'A', 'space', 'del', 'nothing')
            confidence: Confidence score (0-1)
            
        Returns:
            Tuple of (sentence_buffer, letter_confirmed):
            - sentence_buffer: Current accumulated sentence
            - letter_confirmed: Whether a letter was just confirmed this frame
        """
        letter_confirmed = False
        
        # Check if confidence meets threshold
        if confidence < self.confidence_threshold:
            self._reset_tracking()
            return self.sentence_buffer, letter_confirmed
        
        # Check if this is the same letter as we're tracking
        if predicted_label == self.current_letter:
            self.stability_counter += 1
            
            if self.stability_counter >= self.stability_frames:
                if predicted_label != self.last_confirmed_letter:
                    letter_confirmed = self._confirm_letter(predicted_label)
                    self.last_confirmed_letter = predicted_label
        else:
            self.current_letter = predicted_label
            self.stability_counter = 1
            
        return self.sentence_buffer, letter_confirmed
    
    def _reset_tracking(self):
        """Reset tracking state for low confidence predictions."""
        self.current_letter = None
        self.stability_counter = 0
    
    def _confirm_letter(self, letter: str) -> bool:
        """
        Confirm a letter and update the sentence buffer.
        
        Args:
            letter: The letter to confirm
            
        Returns:
            True if the letter was processed, False otherwise
        """
        if letter == 'space':
            self.sentence_buffer += ' '
        elif letter == 'del':
            if self.sentence_buffer:
                self.sentence_buffer = self.sentence_buffer[:-1]
        elif letter == 'nothing':
            pass
        else:
            self.sentence_buffer += letter
        
        self.stability_counter = 0
        return True
    
    def clear(self):
        """Clear the sentence buffer and reset state."""
        self.sentence_buffer = ""
        self.current_letter = None
        self.stability_counter = 0
        self.last_confirmed_letter = None
    
    def get_sentence(self) -> str:
        """Get the current sentence buffer."""
        return self.sentence_buffer
    
    def get_stability_progress(self) -> float:
        """
        Get the stability progress as a ratio (0-1).
        
        Returns:
            Ratio of current stability counter to required frames
        """
        if self.stability_frames == 0:
            return 1.0
        return min(self.stability_counter / self.stability_frames, 1.0)


if __name__ == '__main__':
    print("WordBuilder module loaded successfully")
