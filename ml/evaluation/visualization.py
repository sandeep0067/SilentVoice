"""
Visualization module for evaluation metrics.

Provides plotting functions for confusion matrices, ROC curves, and
misclassified sample analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = False,
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping confusion matrix plot")
        return

    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, nan=0.0)
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curves(
    evaluator,
    output_path: str,
    title: str = 'ROC Curves (One-vs-Rest)'
) -> None:
    """
    Plot ROC curves for each class.

    Args:
        evaluator: EvaluationMetrics object with probability data
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping ROC curves")
        return
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available, skipping ROC curves")
        return

    if len(evaluator.all_probabilities) == 0:
        print("No probability data available, skipping ROC curves")
        return

    probabilities = np.array(evaluator.all_probabilities)
    labels = np.array(evaluator.all_labels)
    num_classes = evaluator.num_classes
    class_names = evaluator.class_names

    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        class_proba = probabilities[:, i]

        if len(np.unique(binary_labels)) > 1:
            fpr, tpr, _ = roc_curve(binary_labels, class_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
            )

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {output_path}")


def plot_per_class_metrics(
    metrics: Dict,
    output_path: str,
    title: str = 'Per-Class Performance Metrics'
) -> None:
    """
    Plot per-class metrics as a bar chart.

    Args:
        metrics: Metrics dictionary from EvaluationMetrics
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping per-class metrics plot")
        return

    class_names = metrics['class_names']
    per_class = metrics['per_class_metrics']

    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    precision = per_class['precision']
    recall = per_class['recall']
    f1 = per_class['f1']
    accuracy = per_class['accuracy']

    bars1 = ax.bar(x - width*1.5, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x - width*0.5, recall, width, label='Recall', color='forestgreen')
    bars3 = ax.bar(x + width*0.5, f1, width, label='F1-Score', color='darkorange')
    bars4 = ax.bar(x + width*1.5, accuracy, width, label='Accuracy', color='crimson')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {output_path}")


def plot_misclassified_distribution(
    misclassified: List[Dict],
    output_path: str,
    title: str = 'Misclassification Distribution'
) -> None:
    """
    Plot distribution of misclassifications.

    Args:
        misclassified: List of misclassified sample info
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping misclassification distribution plot")
        return

    if not misclassified:
        print("No misclassified samples to plot")
        return

    # Count misclassifications per true class
    true_class_counts = {}
    pred_class_counts = {}
    confusion_counts = {}

    for sample in misclassified:
        true_class = sample['true_name']
        pred_class = sample['predicted_name']
        pair = f"{true_class} -> {pred_class}"

        true_class_counts[true_class] = true_class_counts.get(true_class, 0) + 1
        pred_class_counts[pred_class] = pred_class_counts.get(pred_class, 0) + 1
        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Misclassifications by true class
    if true_class_counts:
        axes[0].bar(true_class_counts.keys(), true_class_counts.values(), color='coral')
        axes[0].set_xlabel('True Class', fontsize=12)
        axes[0].set_ylabel('Number of Misclassifications', fontsize=12)
        axes[0].set_title('Misclassifications by True Class', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Top confusion pairs
    if confusion_counts:
        sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        pairs, counts = zip(*sorted_pairs)
        axes[1].barh(range(len(pairs)), counts, color='steelblue')
        axes[1].set_yticks(range(len(pairs)))
        axes[1].set_yticklabels(pairs, fontsize=9)
        axes[1].set_xlabel('Count', fontsize=12)
        axes[1].set_title('Top 15 Confusion Pairs', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Misclassification distribution saved to {output_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str,
    title: str = 'Training History'
) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with train_loss, train_acc, val_loss, val_acc
        output_path: Path to save the plot
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping training history plot")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {output_path}")


def generate_evaluation_report(
    metrics: Dict,
    output_dir: str,
    plot_confusion: bool = True,
    plot_roc: bool = True,
    plot_per_class: bool = True,
    plot_misclassified: bool = True,
    misclassified_samples: Optional[List[Dict]] = None
) -> None:
    """
    Generate comprehensive evaluation report with all plots.

    Args:
        metrics: Metrics dictionary from EvaluationMetrics
        output_dir: Directory to save all plots
        plot_confusion: Whether to plot confusion matrix
        plot_roc: Whether to plot ROC curves
        plot_per_class: Whether to plot per-class metrics
        plot_misclassified: Whether to plot misclassification analysis
        misclassified_samples: Optional list of misclassified samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if plot_confusion:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            metrics['class_names'],
            str(output_path / 'confusion_matrix.png'),
            normalize=False,
            title='Confusion Matrix'
        )
        
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            metrics['class_names'],
            str(output_path / 'confusion_matrix_normalized.png'),
            normalize=True,
            title='Normalized Confusion Matrix'
        )

    if plot_per_class:
        plot_per_class_metrics(
            metrics,
            str(output_path / 'per_class_metrics.png'),
            title='Per-Class Performance Metrics'
        )

    if plot_misclassified and misclassified_samples:
        plot_misclassified_distribution(
            misclassified_samples,
            str(output_path / 'misclassification_distribution.png'),
            title='Misclassification Distribution'
        )

    print(f"Evaluation plots saved to {output_dir}")
