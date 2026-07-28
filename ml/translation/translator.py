"""
Main translator module for ISL recognition.

Combines label mapping and prediction objects to provide translation
from model predictions to ISL words/phrases with confidence thresholding.
"""

import logging
from typing import Optional, Dict, List, Union
from pathlib import Path
import numpy as np

from ml.translation.label_mapping import LabelMapping
from ml.translation.prediction import TranslationResult, BatchTranslationResult, PredictionStatus


logger = logging.getLogger(__name__)


class TranslatorConfig:
    """Configuration for translator."""
    
    def __init__(
        self,
        default_language: str = "en",
        confidence_threshold: float = 0.5,
        low_confidence_threshold: float = 0.3,
        return_alternatives: bool = False,
        include_metadata: bool = True
    ):
        """
        Initialize translator configuration.
        
        Args:
            default_language: Default language for translations
            confidence_threshold: Minimum confidence for valid predictions
            low_confidence_threshold: Threshold for low confidence status
            return_alternatives: Whether to return alternative translations
            include_metadata: Whether to include metadata in results
        """
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.return_alternatives = return_alternatives
        self.include_metadata = include_metadata


class ISLTranslator:
    """
    Main translator for ISL recognition.
    
    Independent from ML model - only handles translation logic.
    """
    
    def __init__(
        self,
        label_mapping: Optional[LabelMapping] = None,
        config: Optional[TranslatorConfig] = None
    ):
        """
        Initialize ISL translator.
        
        Args:
            label_mapping: LabelMapping instance (creates empty if None)
            config: TranslatorConfig instance (uses defaults if None)
        """
        self.label_mapping = label_mapping or LabelMapping()
        self.config = config or TranslatorConfig()
        
        logger.info(f"Initialized ISLTranslator with {self.label_mapping.get_num_classes()} classes")
        logger.info(f"Default language: {self.config.default_language}")
        logger.info(f"Confidence threshold: {self.config.confidence_threshold}")
    
    def translate_single(
        self,
        class_id: int,
        confidence: float,
        probabilities: Optional[Dict[int, float]] = None,
        language: Optional[str] = None,
        inference_time_ms: Optional[float] = None
    ) -> TranslationResult:
        """
        Translate a single prediction to ISL word/phrase.
        
        Args:
            class_id: Predicted class ID
            confidence: Prediction confidence (0-1)
            probabilities: Full probability distribution
            language: Target language (uses default if None)
            inference_time_ms: Inference time in milliseconds
            
        Returns:
            TranslationResult with translation and status
        """
        # Get label entry
        label_entry = self.label_mapping.get_label(class_id)
        
        # Determine status based on confidence
        if confidence >= self.config.confidence_threshold:
            status = PredictionStatus.VALID
        elif confidence >= self.config.low_confidence_threshold:
            status = PredictionStatus.LOW_CONFIDENCE
        else:
            status = PredictionStatus.UNKNOWN
        
        # Get translation
        lang = language or self.config.default_language
        
        if label_entry:
            translated_word = label_entry.word
            translated_phrase = label_entry.get_translation(lang)
            alternative_translations = []
            
            if self.config.return_alternatives:
                alternative_translations = [
                    label_entry.translations.get(lang_code, label_entry.word)
                    for lang_code in label_entry.translations.keys()
                    if lang_code != lang
                ]
            
            metadata = {}
            if self.config.include_metadata:
                metadata = {
                    'category': label_entry.category,
                    'description': label_entry.description,
                    'class_id': class_id
                }
        else:
            translated_word = f"Unknown_{class_id}"
            translated_phrase = f"Unknown class {class_id}"
            alternative_translations = []
            metadata = {'class_id': class_id} if self.config.include_metadata else {}
        
        # Create result
        result = TranslationResult(
            class_id=class_id,
            confidence=confidence,
            probabilities=probabilities,
            translated_word=translated_word,
            translated_phrase=translated_phrase,
            language=lang,
            status=status,
            inference_time_ms=inference_time_ms,
            alternative_translations=alternative_translations,
            metadata=metadata
        )
        
        return result
    
    def translate_batch(
        self,
        predictions: List[Dict[str, Union[int, float]]],
        language: Optional[str] = None
    ) -> BatchTranslationResult:
        """
        Translate a batch of predictions.
        
        Args:
            predictions: List of prediction dictionaries with 'class_id' and 'confidence'
            language: Target language (uses default if None)
            
        Returns:
            BatchTranslationResult with all translations
        """
        batch_result = BatchTranslationResult()
        
        for pred in predictions:
            class_id = pred.get('class_id')
            confidence = pred.get('confidence', 0.0)
            probabilities = pred.get('probabilities')
            inference_time_ms = pred.get('inference_time_ms')
            
            result = self.translate_single(
                class_id=class_id,
                confidence=confidence,
                probabilities=probabilities,
                language=language,
                inference_time_ms=inference_time_ms
            )
            
            batch_result.add_result(result)
        
        return batch_result
    
    def translate_with_threshold(
        self,
        class_id: int,
        confidence: float,
        language: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> Optional[TranslationResult]:
        """
        Translate with custom confidence threshold.
        
        Args:
            class_id: Predicted class ID
            confidence: Prediction confidence
            language: Target language
            custom_threshold: Custom confidence threshold (uses default if None)
            
        Returns:
            TranslationResult if above threshold, None otherwise
        """
        threshold = custom_threshold or self.config.confidence_threshold
        
        if confidence < threshold:
            logger.debug(f"Prediction below threshold: {confidence:.3f} < {threshold:.3f}")
            return None
        
        return self.translate_single(class_id, confidence, language=language)
    
    def get_top_k_translations(
        self,
        probabilities: Dict[int, float],
        k: int = 5,
        language: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Get top-k translations from probability distribution.
        
        Args:
            probabilities: Dictionary of class_id -> probability
            k: Number of top predictions to return
            language: Target language
            
        Returns:
            List of TranslationResults sorted by probability
        """
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_k = sorted_probs[:k]
        
        # Translate each
        results = []
        for class_id, prob in top_k:
            result = self.translate_single(
                class_id=class_id,
                confidence=prob,
                probabilities=probabilities,
                language=language
            )
            results.append(result)
        
        return results
    
    def get_translation_by_word(self, word: str, language: Optional[str] = None) -> Optional[TranslationResult]:
        """
        Get translation by looking up word.
        
        Args:
            word: Word to lookup
            language: Target language
            
        Returns:
            TranslationResult or None if word not found
        """
        class_id = self.label_mapping.get_class_id(word)
        
        if class_id is None:
            logger.warning(f"Word not found in mapping: {word}")
            return None
        
        # Create result with maximum confidence (since it's a direct lookup)
        return self.translate_single(
            class_id=class_id,
            confidence=1.0,
            language=language
        )
    
    def get_category_translations(
        self,
        category: str,
        language: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Get all translations for a category.
        
        Args:
            category: Category name
            language: Target language
            
        Returns:
            List of TranslationResults for the category
        """
        label_entries = self.label_mapping.get_category_labels(category)
        
        results = []
        for entry in label_entries:
            result = self.translate_single(
                class_id=entry.class_id,
                confidence=1.0,
                language=language
            )
            results.append(result)
        
        return results
    
    def update_config(self, **kwargs) -> None:
        """
        Update translator configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """
        Save label mapping to file.
        
        Args:
            filepath: Path to save file
        """
        self.label_mapping.save_to_file(filepath)
    
    def load_mapping(self, filepath: Union[str, Path]) -> None:
        """
        Load label mapping from file.
        
        Args:
            filepath: Path to load file from
        """
        self.label_mapping = LabelMapping.load_from_file(filepath)
        logger.info(f"Loaded mapping with {self.label_mapping.get_num_classes()} classes")
    
    def get_statistics(self) -> Dict[str, Union[int, float, str]]:
        """
        Get translator statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_classes': self.label_mapping.get_num_classes(),
            'default_language': self.config.default_language,
            'confidence_threshold': self.config.confidence_threshold,
            'low_confidence_threshold': self.config.low_confidence_threshold,
            'num_categories': len(self.label_mapping.get_all_categories()),
            'return_alternatives': self.config.return_alternatives
        }
