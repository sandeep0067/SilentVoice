"""
Evaluation package for model assessment.
"""

from ml.evaluation.metrics import EvaluationMetrics, evaluate_model
from ml.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_metrics,
    plot_misclassified_distribution,
    plot_training_history,
    generate_evaluation_report
)

__all__ = [
    'EvaluationMetrics',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_per_class_metrics',
    'plot_misclassified_distribution',
    'plot_training_history',
    'generate_evaluation_report',
]
