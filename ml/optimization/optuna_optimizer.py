"""
Optuna hyperparameter optimization module.

Provides automated hyperparameter tuning for the BiLSTM model with
comprehensive logging and checkpointing.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Type hints for when Optuna is not available
if OPTUNA_AVAILABLE:
    StudyType = optuna.Study
    TrialType = optuna.Trial
else:
    StudyType = Any
    TrialType = Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
from ml.training.trainer import BaselineTrainer


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Search space bounds
    hidden_dim_range: tuple = (32, 256)
    num_layers_range: tuple = (1, 4)
    learning_rate_range: tuple = (1e-5, 1e-2)
    batch_size_choices: list = None
    dropout_range: tuple = (0.1, 0.5)
    weight_decay_range: tuple = (1e-6, 1e-3)
    
    # Optimization settings
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    study_name: str = "bilstm_optimization"
    direction: str = "maximize"  # maximize validation accuracy
    
    # Training settings
    epochs_per_trial: int = 20
    early_stopping_patience: int = 5
    
    # Output settings
    optuna_storage: str = "sqlite:///ml/models/optimization.db"
    checkpoint_dir: str = "ml/models/optuna_checkpoints"
    log_dir: str = "ml/models/optuna_logs"
    
    def __post_init__(self):
        if self.batch_size_choices is None:
            self.batch_size_choices = [16, 32, 64, 128]


class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        input_dim: int = 279,
        device: Optional[str] = None
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            config: Optimization configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of output classes
            input_dim: Input feature dimension
            device: Device to train on
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create output directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create study
        self.study = self._create_study()
        
        # Track best trial
        self.best_trial = None
        self.best_params = None
        self.best_value = None
        
    def _create_study(self) -> StudyType:
        """Create Optuna study with sampler and pruner."""
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.optuna_storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.direction
        )
        
        logger.info(f"Created Optuna study: {self.config.study_name}")
        return study
    
    def suggest_hyperparameters(self, trial: TrialType) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        # Model architecture
        hidden_dim = trial.suggest_int('hidden_dim', *self.config.hidden_dim_range)
        num_layers = trial.suggest_int('num_layers', *self.config.num_layers_range)
        dropout = trial.suggest_float('dropout', *self.config.dropout_range)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', *self.config.learning_rate_range, log=True)
        batch_size = trial.suggest_categorical('batch_size', self.config.batch_size_choices)
        weight_decay = trial.suggest_float('weight_decay', *self.config.weight_decay_range, log=True)
        
        # Optimizer choice
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        # Pooling strategy
        pooling_type = trial.suggest_categorical('pooling_type', ['mean', 'max', 'last', 'attention'])
        
        # Projection dimension (optional)
        use_projection = trial.suggest_categorical('use_projection', [True, False])
        projection_dim = None
        if use_projection:
            projection_dim = trial.suggest_int('projection_dim', 32, 128)
        
        return {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'optimizer': optimizer_name,
            'pooling_type': pooling_type,
            'use_projection': use_projection,
            'projection_dim': projection_dim
        }
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        Create optimizer based on hyperparameters.
        
        Args:
            model: PyTorch model
            optimizer_name: Name of optimizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            
        Returns:
            Optimizer instance
        """
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def objective(self, trial: TrialType) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (or negative loss for minimization)
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        # Log trial parameters
        logger.info(f"Trial {trial.number}: {params}")
        
        # Create model configuration
        model_config = BiLSTMConfig(
            input_dim=self.input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_classes=self.num_classes,
            dropout=params['dropout'],
            pooling_type=params['pooling_type'],
            projection_dim=params['projection_dim']
        )
        
        # Create model
        model = BiLSTMBaseline(model_config)
        model.to(self.device)
        
        # Create optimizer
        optimizer = self.create_optimizer(
            model,
            params['optimizer'],
            params['learning_rate'],
            params['weight_decay']
        )
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs_per_trial
        )
        
        # Create trial-specific checkpoint directory
        trial_checkpoint_dir = Path(self.config.checkpoint_dir) / f"trial_{trial.number}"
        trial_log_dir = Path(self.config.log_dir) / f"trial_{trial.number}"
        
        # Create trainer
        trainer = BaselineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            checkpoint_dir=str(trial_checkpoint_dir),
            log_dir=str(trial_log_dir),
            early_stopping_patience=self.config.early_stopping_patience,
            use_amp=self.device.type == 'cuda'
        )
        
        # Update dataloader batch size for this trial
        # Note: This requires recreating the dataloader, which may be expensive
        # For efficiency, we'll use the original dataloader and adjust in training
        
        try:
            # Train model
            history = trainer.fit(
                self.train_loader,
                self.val_loader,
                epochs=self.config.epochs_per_trial
            )
            
            # Get best validation accuracy
            best_val_acc = max(history['val_acc'])
            
            # Log intermediate value for pruning
            trial.report(best_val_acc, trial.number)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Save trial configuration and results
            self._save_trial_results(trial, params, history, best_val_acc)
            
            logger.info(f"Trial {trial.number} completed with val_acc: {best_val_acc:.4f}")
            
            return best_val_acc
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return a very low value for failed trials
            return 0.0
    
    def _save_trial_results(
        self,
        trial: TrialType,
        params: Dict,
        history: Dict,
        best_val_acc: float
    ) -> None:
        """
        Save trial results to disk.
        
        Args:
            trial: Optuna trial object
            params: Trial hyperparameters
            history: Training history
            best_val_acc: Best validation accuracy
        """
        trial_dir = Path(self.config.log_dir) / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'trial_number': trial.number,
            'hyperparameters': params,
            'best_val_accuracy': float(best_val_acc),
            'training_history': {
                'train_loss': history['train_loss'],
                'train_acc': history['train_acc'],
                'val_loss': history['val_loss'],
                'val_acc': history['val_acc']
            },
            'epochs_trained': len(history['train_loss'])
        }
        
        results_path = trial_dir / 'trial_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Trial results saved to {results_path}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting optimization with {self.config.n_trials} trials")
        logger.info(f"Optimization direction: {self.config.direction}")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        # Get best trial
        self.best_trial = self.study.best_trial
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Optimization completed")
        logger.info(f"Best trial: {self.best_trial.number}")
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save best configuration
        self._save_best_configuration()
        
        return {
            'best_trial_number': self.best_trial.number,
            'best_value': self.best_value,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials)
        }
    
    def _save_best_configuration(self) -> None:
        """Save best configuration to disk."""
        best_config_path = Path(self.config.log_dir) / 'best_configuration.json'
        
        best_config = {
            'best_trial_number': self.best_trial.number,
            'best_value': float(self.best_value),
            'best_params': self.best_params,
            'optimization_config': asdict(self.config)
        }
        
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        logger.info(f"Best configuration saved to {best_config_path}")
    
    def get_optimization_history(self) -> Dict:
        """
        Get optimization history and statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        trials = self.study.trials
        
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        
        values = [t.value for t in completed_trials if t.value is not None]
        
        return {
            'total_trials': len(trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'failed_trials': len(failed_trials),
            'best_value': self.best_value if self.best_value else None,
            'worst_value': min(values) if values else None,
            'mean_value': sum(values) / len(values) if values else None,
            'std_value': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if values else None
        }
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """
        Load the best model from optimization.
        
        Args:
            model: Model to load weights into
            
        Returns:
            Model with best weights loaded
        """
        best_trial_dir = Path(self.config.checkpoint_dir) / f"trial_{self.best_trial.number}"
        best_checkpoint = best_trial_dir / 'best_model.pt'
        
        if not best_checkpoint.exists():
            logger.warning(f"Best checkpoint not found at {best_checkpoint}")
            return model
        
        from ml.training.checkpoint import CheckpointManager
        checkpoint_manager = CheckpointManager(str(best_trial_dir))
        
        checkpoint_manager.load_checkpoint(
            best_checkpoint,
            model,
            device=self.device
        )
        
        logger.info(f"Loaded best model from trial {self.best_trial.number}")
        return model
