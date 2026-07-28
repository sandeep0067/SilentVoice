"""
Service layer for inference logic.

Handles model loading, prediction, and translation logic separate from API routes.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import torch

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.translation import ISLTranslator, LabelMapping, TranslatorConfig
from api.config import settings


logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for model inference.
    
    Handles model loading and prediction logic independently from API layer.
    """
    
    def __init__(self):
        """Initialize inference service."""
        self.model: Optional[BiLSTMBaseline] = None
        self.device: Optional[torch.device] = None
        self.is_loaded: bool = False
        self.model_config: Optional[Dict[str, Any]] = None
        
        # Translation service
        self.translator: Optional[ISLTranslator] = None
        self.label_mapping: Optional[LabelMapping] = None
        
        logger.info("InferenceService initialized")
    
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = settings.get_model_path()
            
            if not model_path.exists():
                logger.error(f"Model checkpoint not found: {model_path}")
                return False
            
            logger.info(f"Loading model from {model_path}")
            
            # Setup device
            self.device = torch.device(settings.model_device if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model configuration
            if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
                config_dict = checkpoint['model_config']
                self.model_config = config_dict
                model_config = BiLSTMConfig.from_dict(config_dict)
                logger.info("Loaded model configuration from checkpoint")
            else:
                # Use default config
                model_config = BiLSTMConfig(
                    input_dim=settings.feature_dim,
                    num_classes=25  # Default, will be updated if available
                )
                self.model_config = model_config.to_dict()
                logger.info("Using default model configuration")
            
            # Create model
            self.model = BiLSTMBaseline(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Initialize translation service
            self._init_translation_service()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def _init_translation_service(self) -> None:
        """Initialize translation service."""
        try:
            # Try to load label mapping from file
            label_mapping_path = settings.get_label_mapping_path()
            
            if label_mapping_path and label_mapping_path.exists():
                self.label_mapping = LabelMapping.load_from_file(label_mapping_path)
                logger.info(f"Loaded label mapping from {label_mapping_path}")
            else:
                # Create sample mapping
                self.label_mapping = LabelMapping(default_language=settings.default_language)
                self.label_mapping.create_sample_mapping(num_classes=self.model_config.get('num_classes', 25))
                logger.info("Created sample label mapping")
            
            # Create translator
            translator_config = TranslatorConfig(
                default_language=settings.default_language,
                confidence_threshold=settings.confidence_threshold
            )
            self.translator = ISLTranslator(
                label_mapping=self.label_mapping,
                config=translator_config
            )
            
            logger.info("Translation service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
    
    def predict(
        self,
        features: List[List[float]],
        return_probabilities: bool = False,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction from features.
        
        Args:
            features: Feature vectors (sequence of frames)
            return_probabilities: Whether to return full probability distribution
            language: Language for translation
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded or self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Convert to tensor
            features_array = np.array(features, dtype=np.float32)
            features_tensor = torch.FloatTensor(features_array).to(self.device)
            
            # Add batch dimension
            if features_tensor.ndim == 2:
                features_tensor = features_tensor.unsqueeze(0)  # (1, seq_len, feature_dim)
            
            # Inference
            with torch.no_grad():
                logits = self.model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            probabilities_array = probabilities.cpu().numpy()[0]
            
            # Get label
            if self.translator:
                translation = self.translator.translate_single(
                    class_id=predicted_class,
                    confidence=confidence_score,
                    language=language
                )
                predicted_label = translation.translated_phrase
            else:
                predicted_label = f"Class_{predicted_class}"
            
            # Prepare result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence_score,
                'inference_time_ms': inference_time
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    i: float(prob) for i, prob in enumerate(probabilities_array)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_batch(
        self,
        sequences: List[List[List[float]]],
        return_probabilities: bool = False,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            sequences: List of feature sequences
            return_probabilities: Whether to return probability distributions
            language: Language for translation
            
        Returns:
            Dictionary with batch prediction results
        """
        if not self.is_loaded or self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            start_time = time.time()
            
            predictions = []
            successful = 0
            
            for features in sequences:
                result = self.predict(features, return_probabilities, language)
                predictions.append(result)
                if result['success']:
                    successful += 1
            
            total_time = (time.time() - start_time) * 1000  # ms
            avg_time = total_time / len(sequences) if sequences else 0
            
            return {
                'success': True,
                'predictions': predictions,
                'total_predictions': len(predictions),
                'successful_predictions': successful,
                'total_inference_time_ms': total_time,
                'average_inference_time_ms': avg_time
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def translate(
        self,
        class_id: int,
        confidence: float = 1.0,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate class ID to word/phrase.
        
        Args:
            class_id: Class ID to translate
            confidence: Confidence score
            language: Target language
            
        Returns:
            Dictionary with translation result
        """
        if not self.translator:
            return {
                'success': False,
                'error': 'Translation service not initialized'
            }
        
        try:
            result = self.translator.translate_single(
                class_id=class_id,
                confidence=confidence,
                language=language
            )
            
            return {
                'success': True,
                'class_id': result.class_id,
                'translated_word': result.translated_word,
                'translated_phrase': result.translated_phrase,
                'language': result.language,
                'confidence': result.confidence,
                'status': result.status.value
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded or self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            num_classes = self.model_config.get('num_classes', 0)
            
            # Get class names if available
            class_names = None
            if self.label_mapping:
                class_names = [
                    self.label_mapping.get_word(i)
                    for i in range(num_classes)
                ]
            
            return {
                'success': True,
                'model_name': 'BiLSTM Baseline',
                'model_version': '1.0.0',
                'num_classes': num_classes,
                'feature_dim': self.model_config.get('input_dim', settings.feature_dim),
                'architecture': 'BiLSTM',
                'device': str(self.device),
                'checkpoint_path': str(settings.get_model_path()),
                'class_names': class_names,
                'model_config': self.model_config
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.is_loaded
    
    def get_device(self) -> str:
        """Get current device."""
        return str(self.device) if self.device else "unknown"


# Global inference service instance
inference_service = InferenceService()
