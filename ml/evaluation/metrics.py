"""
Evaluation metrics module for model assessment.

Provides comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrix, and per-class metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
        roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics calculator.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize evaluation metrics.

        Args:
            num_classes: Number of classes in the dataset
            class_names: Optional list of class names for better visualization
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> None:
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: Predicted class indices
            labels: Ground truth class indices
            probabilities: Optional prediction probabilities for ROC curves
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        self.all_predictions.extend(predictions.flatten().tolist())
        self.all_labels.extend(labels.flatten().tolist())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.tolist())

    def compute_metrics(self) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary containing all computed metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for evaluation metrics")

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision = precision_score(labels, predictions, average=None, zero_division=0)
        recall = recall_score(labels, predictions, average=None, zero_division=0)
        f1 = f1_score(labels, predictions, average=None, zero_division=0)

        # Average metrics
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'accuracy': per_class_accuracy.tolist()
            },
            'confusion_matrix': cm,
            'class_names': self.class_names
        }

        # ROC AUC if probabilities available
        if len(self.all_probabilities) > 0:
            try:
                probabilities = np.array(self.all_probabilities)
                if probabilities.shape[1] == self.num_classes:
                    # One-vs-rest ROC AUC
                    roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
                    metrics['roc_auc_macro'] = roc_auc
                    
                    # Per-class ROC AUC
                    roc_auc_per_class = []
                    for i in range(self.num_classes):
                        binary_labels = (labels == i).astype(int)
                        class_proba = probabilities[:, i]
                        if len(np.unique(binary_labels)) > 1:
                            class_auc = roc_auc_score(binary_labels, class_proba)
                            roc_auc_per_class.append(class_auc)
                        else:
                            roc_auc_per_class.append(0.0)
                    
                    metrics['per_class_metrics']['roc_auc'] = roc_auc_per_class
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC: {e}")

        return metrics

    def get_classification_report(self) -> str:
        """
        Get sklearn classification report as string.

        Returns:
            Classification report string
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classification report")

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        return classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            zero_division=0
        )

    def get_misclassified_samples(
        self,
        dataset_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Get information about misclassified samples.

        Args:
            dataset_indices: Optional list of dataset indices corresponding to predictions

        Returns:
            List of dictionaries containing misclassification details
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        misclassified_mask = predictions != labels
        misclassified_indices = np.where(misclassified_mask)[0]

        misclassified = []
        for idx in misclassified_indices:
            info = {
                'index': idx,
                'predicted': int(predictions[idx]),
                'predicted_name': self.class_names[predictions[idx]],
                'true': int(labels[idx]),
                'true_name': self.class_names[labels[idx]]
            }
            if dataset_indices is not None and idx < len(dataset_indices):
                info['dataset_index'] = dataset_indices[idx]
            misclassified.append(info)

        return misclassified

    def export_to_csv(self, output_path: str) -> None:
        """
        Export metrics to CSV file.

        Args:
            output_path: Path to save CSV file
        """
        if not PANDAS_AVAILABLE:
            print("pandas not available, skipping CSV export")
            return

        metrics = self.compute_metrics()

        # Create summary metrics DataFrame
        summary_data = {
            'Metric': [
                'Accuracy',
                'Precision (Macro)',
                'Recall (Macro)',
                'F1-Score (Macro)',
                'Precision (Weighted)',
                'Recall (Weighted)',
                'F1-Score (Weighted)'
            ],
            'Value': [
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro'],
                metrics['precision_weighted'],
                metrics['recall_weighted'],
                metrics['f1_weighted']
            ]
        }

        if 'roc_auc_macro' in metrics:
            summary_data['Metric'].append('ROC AUC (Macro)')
            summary_data['Value'].append(metrics['roc_auc_macro'])

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)

        # Create per-class metrics DataFrame
        per_class_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': metrics['per_class_metrics']['precision'],
            'Recall': metrics['per_class_metrics']['recall'],
            'F1-Score': metrics['per_class_metrics']['f1'],
            'Accuracy': metrics['per_class_metrics']['accuracy']
        })

        if 'roc_auc' in metrics['per_class_metrics']:
            per_class_df['ROC AUC'] = metrics['per_class_metrics']['roc_auc']

        per_class_path = str(Path(output_path).parent / 'per_class_metrics.csv')
        per_class_df.to_csv(per_class_path, index=False)

        print(f"Metrics exported to {output_path}")
        print(f"Per-class metrics exported to {per_class_path}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Tuple[EvaluationMetrics, Dict]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional class names

    Returns:
        Tuple of (EvaluationMetrics object, metrics dictionary)
    """
    model.eval()
    evaluator = EvaluationMetrics(num_classes, class_names)

    dataset_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                x, y, lengths = batch
                x, y = x.to(device), y.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)

            if y.ndim > 1:
                y = y.squeeze(-1)

            # Get predictions
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Track dataset indices
            batch_size = x.size(0)
            start_idx = batch_idx * dataloader.batch_size
            dataset_indices.extend(range(start_idx, start_idx + batch_size))

            # Update evaluator
            evaluator.update(predictions, y, probabilities)

    metrics = evaluator.compute_metrics()
    return evaluator, metrics
