"""
Optimized model utilities for faster inference.

Provides model quantization, TorchScript compilation, and other
performance optimizations while maintaining accuracy.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Model optimization utilities.
    """
    
    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: PyTorch model to quantize
            dtype: Quantization dtype (default: qint8)
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying dynamic quantization with dtype={dtype}")
        
        # Quantize linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM},
            dtype=dtype
        )
        
        logger.info("Dynamic quantization complete")
        return quantized_model
    
    @staticmethod
    def quantize_static(
        model: nn.Module, 
        calibration_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> nn.Module:
        """
        Apply static quantization to model.
        
        Args:
            model: PyTorch model to quantize
            calibration_loader: Data loader for calibration
            device: Device to run calibration on
            
        Returns:
            Quantized model
        """
        logger.info("Applying static quantization")
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm' if device.type == 'cpu' else 'qnnpack')
        
        # Prepare model
        model = torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        logger.info("Calibrating model...")
        model.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(device)
                _ = model(batch)
        
        # Convert to quantized
        model = torch.quantization.convert(model, inplace=True)
        logger.info("Static quantization complete")
        
        return model
    
    @staticmethod
    def compile_model(model: nn.Module, mode: str = 'reduce-overhead') -> nn.Module:
        """
        Compile model using torch.compile (PyTorch 2.0+).
        
        Args:
            model: PyTorch model to compile
            mode: Compilation mode ('reduce-overhead', 'max-autotune', 'default')
            
        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
        
        logger.info(f"Compiling model with mode={mode}")
        
        try:
            compiled_model = torch.compile(model, mode=mode)
            logger.info("Model compilation complete")
            return compiled_model
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model
    
    @staticmethod
    def to_torchscript(model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """
        Convert model to TorchScript.
        
        Args:
            model: PyTorch model to convert
            example_input: Example input for tracing
            
        Returns:
            TorchScript model
        """
        logger.info("Converting model to TorchScript")
        
        model.eval()
        
        try:
            # Try tracing first
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            
            logger.info("TorchScript conversion complete (tracing)")
            return traced_model
        except Exception as e:
            logger.warning(f"Tracing failed: {e}, trying scripting...")
            
            try:
                scripted_model = torch.jit.script(model)
                logger.info("TorchScript conversion complete (scripting)")
                return scripted_model
            except Exception as e2:
                logger.error(f"Scripting failed: {e2}")
                raise
    
    @staticmethod
    def prune_model(
        model: nn.Module, 
        amount: float = 0.2,
        pruning_method: str = 'l1_unstructured'
    ) -> nn.Module:
        """
        Prune model to reduce size and improve speed.
        
        Args:
            model: PyTorch model to prune
            amount: Fraction of parameters to prune (0.0-1.0)
            pruning_method: Pruning method ('l1_unstructured', 'l1_structured')
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        logger.info(f"Pruning model with amount={amount}, method={pruning_method}")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif isinstance(module, nn.LSTM):
                prune.l1_unstructured(module, name='weight_ih_l0', amount=amount)
                prune.l1_unstructured(module, name='weight_hh_l0', amount=amount)
        
        logger.info("Model pruning complete")
        return model
    
    @staticmethod
    def apply_mixed_precision(model: nn.Module) -> nn.Module:
        """
        Enable automatic mixed precision (AMP) for faster GPU inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model ready for AMP
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, mixed precision not applicable")
            return model
        
        logger.info("Enabling automatic mixed precision")
        # AMP is used during inference with torch.cuda.amp.autocast()
        return model
    
    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        use_quantization: bool = True,
        use_torchscript: bool = True,
        use_compile: bool = True,
        device: torch.device = torch.device('cpu')
    ) -> nn.Module:
        """
        Apply comprehensive optimizations for inference.
        
        Args:
            model: PyTorch model to optimize
            checkpoint_path: Path to checkpoint (if loading from file)
            use_quantization: Whether to apply quantization
            use_torchscript: Whether to convert to TorchScript
            use_compile: Whether to compile with torch.compile
            device: Target device
            
        Returns:
            Optimized model
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive model optimization")
        logger.info("=" * 60)
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_config' in checkpoint and checkpoint['model_config']:
                config = BiLSTMConfig.from_dict(checkpoint['model_config'])
            else:
                config = BiLSTMConfig()
            
            model = BiLSTMBaseline(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Apply quantization
        if use_quantization and device.type == 'cpu':
            logger.info("Applying dynamic quantization")
            model = ModelOptimizer.quantize_dynamic(model)
        
        # Apply torch.compile
        if use_compile:
            model = ModelOptimizer.compile_model(model)
        
        # Convert to TorchScript
        if use_torchscript:
            # Create example input
            example_input = torch.randn(1, 30, 279).to(device)
            model = ModelOptimizer.to_torchscript(model, example_input)
        
        logger.info("=" * 60)
        logger.info("Model optimization complete")
        logger.info("=" * 60)
        
        return model
    
    @staticmethod
    def save_optimized_model(
        model: nn.Module,
        output_path: str,
        optimization_info: Dict[str, Any]
    ):
        """
        Save optimized model with metadata.
        
        Args:
            model: Optimized model
            output_path: Path to save model
            optimization_info: Dictionary with optimization details
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, torch.jit.ScriptModule):
            # Save TorchScript model
            torch.jit.save(model, str(output_path))
            logger.info(f"Saved TorchScript model to {output_path}")
        else:
            # Save state dict with metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimization_info': optimization_info
            }, str(output_path))
            logger.info(f"Saved optimized model to {output_path}")


def benchmark_model(
    model: nn.Module,
    input_shape: tuple = (1, 30, 279),
    num_runs: int = 100,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        device: Device to run on
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'fps': 1000 / np.mean(times)
    }
