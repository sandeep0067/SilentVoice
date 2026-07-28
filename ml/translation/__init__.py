"""
Translation module for ISL recognition.

Provides independent translation layer for converting model predictions
to ISL words/phrases with multi-language support and confidence thresholding.
"""

from ml.translation.prediction import (
    TranslationResult,
    BatchTranslationResult,
    PredictionStatus
)
from ml.translation.label_mapping import (
    LabelMapping,
    LabelEntry
)
from ml.translation.translator import (
    ISLTranslator,
    TranslatorConfig
)

__all__ = [
    'TranslationResult',
    'BatchTranslationResult',
    'PredictionStatus',
    'LabelMapping',
    'LabelEntry',
    'ISLTranslator',
    'TranslatorConfig',
]
