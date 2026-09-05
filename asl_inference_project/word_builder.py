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
    
    # Space detection configuration
    DEFAULT_SPACE_STABILITY_FRAMES = 2
    DEFAULT_SPACE_CONFIDENCE_THRESHOLD = 0.4
    DEFAULT_NO_HAND_THRESHOLD = 15  # frames
    
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
        self.word_count = 0
        
        # Space detection improvements
        self.space_stability_frames = self.DEFAULT_SPACE_STABILITY_FRAMES
        self.space_confidence_threshold = self.DEFAULT_SPACE_CONFIDENCE_THRESHOLD
        self.no_hand_frames = 0
        self.no_hand_threshold = self.DEFAULT_NO_HAND_THRESHOLD
        
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
        
        # Special handling for space with lower thresholds
        if predicted_label == 'space':
            if confidence >= self.space_confidence_threshold:
                if predicted_label == self.current_letter:
                    self.stability_counter += 1
                    if self.stability_counter >= self.space_stability_frames:
                        if predicted_label != self.last_confirmed_letter:
                            letter_confirmed = self._confirm_letter(predicted_label)
                            self.last_confirmed_letter = predicted_label
                else:
                    self.current_letter = predicted_label
                    self.stability_counter = 1
            return self.sentence_buffer, letter_confirmed
        
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
    
    def update_no_hand(self) -> Tuple[str, bool]:
        """
        Update when no hand is detected (auto-space after threshold).
        
        Returns:
            Tuple of (sentence_buffer, space_added):
            - sentence_buffer: Current accumulated sentence
            - space_added: Whether a space was just added
        """
        space_added = False
        self.no_hand_frames += 1
        
        # Auto-space after threshold if we have content
        if (self.no_hand_frames >= self.no_hand_threshold and 
            self.sentence_buffer and 
            self.sentence_buffer[-1] != ' '):
            self.sentence_buffer += ' '
            self.word_count += 1
            self.last_confirmed_letter = ' '
            space_added = True
            self.no_hand_frames = 0  # Reset counter
        
        return self.sentence_buffer, space_added
    
    def _confirm_letter(self, letter: str) -> bool:
        """
        Confirm a letter and update the sentence buffer.
        
        Args:
            letter: The letter to confirm
            
        Returns:
            True if the letter was processed, False otherwise
        """
        if letter == 'space':
            # Add space and increment word count
            self.sentence_buffer += ' '
            self.word_count += 1
        elif letter == 'del':
            # Remove last character if buffer is not empty
            if self.sentence_buffer:
                removed_char = self.sentence_buffer[-1]
                self.sentence_buffer = self.sentence_buffer[:-1]
                # Update word count if we removed a space
                if removed_char == ' ':
                    self.word_count = max(0, self.word_count - 1)
        elif letter == 'nothing':
            # Do nothing for 'nothing' class
            pass
        else:
            # Regular letter - append to buffer
            self.sentence_buffer += letter
        
        # Reset stability counter after confirming
        self.stability_counter = 0
        return True
    
    def clear(self):
        """Clear the sentence buffer and reset state."""
        self.sentence_buffer = ""
        self.current_letter = None
        self.stability_counter = 0
        self.last_confirmed_letter = None
        self.word_count = 0
    
    def get_sentence(self) -> str:
        """Get the current sentence buffer."""
        return self.sentence_buffer
    
    def get_word(self) -> str:
        """Get the current sentence buffer (for backward compatibility)."""
        return self.sentence_buffer
    
    def get_current_letter(self) -> str:
        """Get the currently tracked letter (before confirmation)."""
        return self.current_letter if self.current_letter else ""
    
    def get_stability_progress(self) -> float:
        """
        Get the stability progress as a ratio (0-1).
        
        Returns:
            Ratio of current stability counter to required frames
        """
        if self.stability_frames == 0:
            return 1.0
        return min(self.stability_counter / self.stability_frames, 1.0)
    
    def get_current_word(self) -> str:
        """Get the current word (text since last space)."""
        if not self.sentence_buffer:
            return ""
        # Split by space and get the last word
        words = self.sentence_buffer.split()
        return words[-1] if words else ""
    
    def get_word_count(self) -> int:
        """Get the number of words in the sentence."""
        return self.word_count
    
    def get_stats(self) -> dict:
        """Get statistics about the current sentence."""
        return {
            'sentence': self.sentence_buffer,
            'current_word': self.get_current_word(),
            'word_count': self.word_count,
            'character_count': len(self.sentence_buffer),
            'stability_progress': self.get_stability_progress(),
            'current_letter': self.current_letter if self.current_letter else ""
        }


if __name__ == '__main__':
    # Simple test of the WordBuilder
    print("WordBuilder test")
    print("=" * 50)
    
    wb = WordBuilder(stability_frames=3, confidence_threshold=0.7)
    
    # Test sequence: A A A (should confirm), B B (not enough), B B B (should confirm)
    test_sequence = [
        ('A', 0.8),
        ('A', 0.9),
        ('A', 0.85),
        ('B', 0.8),
        ('B', 0.9),
        ('B', 0.85),
        ('space', 0.9),
        ('space', 0.9),
        ('space', 0.9),
        ('del', 0.9),
        ('del', 0.9),
        ('del', 0.9),
    ]
    
    for label, conf in test_sequence:
        word, confirmed = wb.update(label, conf)
        status = "CONFIRMED" if confirmed else "tracking"
        print(f"Prediction: {label} ({conf:.2f}) -> Word: '{word}' [{status}]")
        print(f"  Current letter: {wb.get_current_letter()}, Stability: {wb.get_stability_progress():.2f}")
        print()
    
    print("=" * 50)
    print("Final word:", wb.get_word())
