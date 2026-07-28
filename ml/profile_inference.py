"""
Comprehensive profiling script for inference pipeline.

Profiles:
- Model loading time and memory usage
- Webcam capture latency
- MediaPipe feature extraction time
- Model inference latency
- Overall FPS
- CPU and memory usage
"""

import time
import gc
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from collections import defaultdict
import json

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory/CPU profiling will be limited")

# Import ML components (with graceful fallbacks)
try:
    from ml.models.bilstm_baseline import BiLSTMBaseline, BiLSTMConfig
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logging.warning("Model components not available")

try:
    from ml.inference.processors.holistic_feature_extractor import HolisticFeatureExtractor, HolisticExtractionConfig
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe components not available")

try:
    from ml.inference.realtime.pipeline import RealtimeInferencePipeline, RealtimeConfig
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logging.warning("Pipeline components not available")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceProfiler:
    """
    Comprehensive profiler for inference pipeline.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.results = defaultdict(list)
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE or not self.process:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not PSUTIL_AVAILABLE or not self.process:
            return 0.0
        return self.process.cpu_percent(interval=0.1)
    
    def profile_model_loading(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Profile model loading time and memory usage.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Dictionary with profiling results
        """
        if not MODEL_AVAILABLE:
            return {'success': False, 'error': 'Model components not available'}
            
        logger.info("Profiling model loading...")
        
        # Force garbage collection before
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        mem_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model
            if 'model_config' in checkpoint and checkpoint['model_config']:
                config = BiLSTMConfig.from_dict(checkpoint['model_config'])
            else:
                config = BiLSTMConfig()
            
            model = BiLSTMBaseline(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            load_time = time.time() - start_time
            mem_after = self.get_memory_usage()
            
            # Get model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            result = {
                'success': True,
                'load_time_seconds': load_time,
                'memory_before_mb': mem_before['rss_mb'],
                'memory_after_mb': mem_after['rss_mb'],
                'memory_increase_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
                'model_size_mb': model_size_mb,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'config': config.to_dict()
            }
            
            logger.info(f"Model loading: {load_time:.3f}s, {model_size_mb:.2f}MB")
            
            del model, checkpoint
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def profile_webcam_capture(self, num_frames: int = 100) -> Dict[str, Any]:
        """
        Profile webcam capture latency.
        
        Args:
            num_frames: Number of frames to capture
            
        Returns:
            Dictionary with profiling results
        """
        logger.info("Profiling webcam capture...")
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open webcam'}
            
            # Warm up
            for _ in range(10):
                cap.read()
            
            capture_times = []
            for i in range(num_frames):
                start = time.time()
                ret, frame = cap.read()
                end = time.time()
                
                if ret:
                    capture_times.append((end - start) * 1000)  # ms
            
            cap.release()
            
            if capture_times:
                capture_times = np.array(capture_times)
                result = {
                    'success': True,
                    'num_frames': len(capture_times),
                    'avg_latency_ms': np.mean(capture_times),
                    'min_latency_ms': np.min(capture_times),
                    'max_latency_ms': np.max(capture_times),
                    'std_latency_ms': np.std(capture_times),
                    'estimated_fps': 1000 / np.mean(capture_times)
                }
                logger.info(f"Webcam capture: {result['avg_latency_ms']:.2f}ms avg, {result['estimated_fps']:.1f} FPS")
                return result
            
            return {'success': False, 'error': 'No frames captured'}
            
        except ImportError:
            return {'success': False, 'error': 'OpenCV not available'}
        except Exception as e:
            logger.error(f"Webcam profiling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def profile_mediapipe_extraction(self, num_frames: int = 50) -> Dict[str, Any]:
        """
        Profile MediaPipe feature extraction.
        
        Args:
            num_frames: Number of frames to process
            
        Returns:
            Dictionary with profiling results
        """
        if not MEDIAPIPE_AVAILABLE:
            return {'success': False, 'error': 'MediaPipe components not available'}
            
        logger.info("Profiling MediaPipe feature extraction...")
        
        try:
            import cv2
            
            # Create feature extractor
            config = HolisticExtractionConfig(
                model_complexity=1,
                enable_hands=True,
                enable_pose=True,
                enable_face=True
            )
            extractor = HolisticFeatureExtractor(config)
            
            # Initialize
            if not extractor.initialize():
                return {'success': False, 'error': 'MediaPipe initialization failed'}
            
            # Capture frames
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open webcam'}
            
            # Warm up
            for _ in range(10):
                ret, frame = cap.read()
                if ret:
                    extractor.process_frame(frame)
            
            extraction_times = []
            feature_dims = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                start = time.time()
                features = extractor.process_frame(frame)
                end = time.time()
                
                if features is not None:
                    extraction_times.append((end - start) * 1000)  # ms
                    if features is not None and len(features) > 0:
                        feature_dims.append(len(features[0]) if isinstance(features[0], (list, np.ndarray)) else len(features))
            
            cap.release()
            extractor.cleanup()
            
            if extraction_times:
                extraction_times = np.array(extraction_times)
                result = {
                    'success': True,
                    'num_frames': len(extraction_times),
                    'avg_time_ms': np.mean(extraction_times),
                    'min_time_ms': np.min(extraction_times),
                    'max_time_ms': np.max(extraction_times),
                    'std_time_ms': np.std(extraction_times),
                    'estimated_fps': 1000 / np.mean(extraction_times),
                    'feature_dim': np.mean(feature_dims) if feature_dims else None
                }
                logger.info(f"MediaPipe extraction: {result['avg_time_ms']:.2f}ms avg, {result['estimated_fps']:.1f} FPS")
                return result
            
            return {'success': False, 'error': 'No features extracted'}
            
        except Exception as e:
            logger.error(f"MediaPipe profiling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def profile_model_inference(self, model: torch.nn.Module, num_runs: int = 100) -> Dict[str, Any]:
        """
        Profile model inference latency.
        
        Args:
            model: PyTorch model
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with profiling results
        """
        if not MODEL_AVAILABLE:
            return {'success': False, 'error': 'Model components not available'}
            
        logger.info("Profiling model inference...")
        
        device = next(model.parameters()).device
        model.eval()
        
        # Create dummy input
        batch_size = 1
        seq_len = 30
        feature_dim = 279
        
        dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_times = []
        mem_usage_before = self.get_memory_usage()
        
        with torch.no_grad():
            for i in range(num_runs):
                start = time.time()
                output = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                inference_times.append((end - start) * 1000)  # ms
        
        mem_usage_after = self.get_memory_usage()
        
        inference_times = np.array(inference_times)
        
        result = {
            'success': True,
            'num_runs': num_runs,
            'avg_time_ms': np.mean(inference_times),
            'min_time_ms': np.min(inference_times),
            'max_time_ms': np.max(inference_times),
            'std_time_ms': np.std(inference_times),
            'p50_ms': np.percentile(inference_times, 50),
            'p95_ms': np.percentile(inference_times, 95),
            'p99_ms': np.percentile(inference_times, 99),
            'estimated_fps': 1000 / np.mean(inference_times),
            'memory_before_mb': mem_usage_before['rss_mb'],
            'memory_after_mb': mem_usage_after['rss_mb'],
            'device': str(device)
        }
        
        logger.info(f"Model inference: {result['avg_time_ms']:.2f}ms avg, {result['estimated_fps']:.1f} FPS")
        
        return result
    
    def profile_full_pipeline(self, num_frames: int = 100) -> Dict[str, Any]:
        """
        Profile the full inference pipeline.
        
        Args:
            num_frames: Number of frames to process
            
        Returns:
            Dictionary with profiling results
        """
        if not PIPELINE_AVAILABLE:
            return {'success': False, 'error': 'Pipeline components not available'}
            
        logger.info("Profiling full inference pipeline...")
        
        try:
            import cv2
            
            # Create pipeline
            config = RealtimeConfig(
                window_size=30,
                confidence_threshold=0.5,
                smoothing_window=5
            )
            
            pipeline = RealtimeInferencePipeline(config)
            
            # Initialize
            if not pipeline.initialize():
                return {'success': False, 'error': 'Pipeline initialization failed'}
            
            # Capture frames
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open webcam'}
            
            # Warm up
            for _ in range(10):
                ret, frame = cap.read()
                if ret:
                    pipeline.process_frame(frame)
            
            # Profile
            frame_times = []
            prediction_times = []
            cpu_usage = []
            memory_usage = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                result = pipeline.process_frame(frame)
                frame_end = time.time()
                
                frame_times.append((frame_end - frame_start) * 1000)
                
                if result:
                    prediction_times.append(result.inference_time_ms if hasattr(result, 'inference_time_ms') else 0)
                
                cpu_usage.append(self.get_cpu_usage())
                memory_usage.append(self.get_memory_usage()['rss_mb'])
            
            cap.release()
            pipeline.cleanup()
            
            if frame_times:
                frame_times = np.array(frame_times)
                result = {
                    'success': True,
                    'num_frames': len(frame_times),
                    'avg_frame_time_ms': np.mean(frame_times),
                    'min_frame_time_ms': np.min(frame_times),
                    'max_frame_time_ms': np.max(frame_times),
                    'std_frame_time_ms': np.std(frame_times),
                    'estimated_fps': 1000 / np.mean(frame_times),
                    'avg_prediction_time_ms': np.mean(prediction_times) if prediction_times else 0,
                    'avg_cpu_percent': np.mean(cpu_usage),
                    'avg_memory_mb': np.mean(memory_usage),
                    'max_memory_mb': np.max(memory_usage)
                }
                logger.info(f"Full pipeline: {result['avg_frame_time_ms']:.2f}ms avg, {result['estimated_fps']:.1f} FPS")
                return result
            
            return {'success': False, 'error': 'No frames processed'}
            
        except Exception as e:
            logger.error(f"Pipeline profiling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_full_profile(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Run complete profiling suite.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Dictionary with all profiling results
        """
        logger.info("=" * 60)
        logger.info("Starting Full Inference Pipeline Profile")
        logger.info("=" * 60)
        
        results = {
            'timestamp': time.time(),
            'system': {
                'cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 'unknown',
                'memory_total_gb': psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 0,
                'python_version': f"{psutil.version_info.major}.{psutil.version_info.minor}.{psutil.version_info.micro}" if PSUTIL_AVAILABLE else 'unknown',
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        }
        
        # Profile model loading
        results['model_loading'] = self.profile_model_loading(checkpoint_path)
        
        # Profile webcam capture
        results['webcam_capture'] = self.profile_webcam_capture()
        
        # Profile MediaPipe extraction
        results['mediapipe_extraction'] = self.profile_mediapipe_extraction()
        
        # Profile model inference (if model loading succeeded)
        if results['model_loading']['success'] and MODEL_AVAILABLE:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                config = BiLSTMConfig.from_dict(checkpoint['model_config']) if checkpoint.get('model_config') else BiLSTMConfig()
                model = BiLSTMBaseline(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                
                results['model_inference'] = self.profile_model_inference(model)
                
                del model, checkpoint
                gc.collect()
                
            except Exception as e:
                logger.error(f"Model inference profiling failed: {e}")
                results['model_inference'] = {'success': False, 'error': str(e)}
        else:
            results['model_inference'] = {'success': False, 'error': 'Model not loaded or components not available'}
        
        # Profile full pipeline
        results['full_pipeline'] = self.profile_full_pipeline()
        
        # Generate recommendations
        results['recommendations'] = self.generate_recommendations(results)
        
        logger.info("=" * 60)
        logger.info("Profiling Complete")
        logger.info("=" * 60)
        
        return results
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate optimization recommendations based on profiling results.
        
        Args:
            results: Profiling results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Model loading recommendations
        if results['model_loading']['success']:
            load_time = results['model_loading']['load_time_seconds']
            if load_time > 5:
                recommendations.append(f"Model loading is slow ({load_time:.2f}s). Consider using torch.compile() or quantization.")
            
            model_size = results['model_loading']['model_size_mb']
            if model_size > 100:
                recommendations.append(f"Model is large ({model_size:.2f}MB). Consider model pruning or quantization.")
        
        # Webcam recommendations
        if results['webcam_capture']['success']:
            fps = results['webcam_capture']['estimated_fps']
            if fps < 30:
                recommendations.append(f"Webcam capture is below 30 FPS ({fps:.1f} FPS). Consider reducing resolution or using a faster camera.")
        
        # MediaPipe recommendations
        if results['mediapipe_extraction']['success']:
            fps = results['mediapipe_extraction']['estimated_fps']
            if fps < 20:
                recommendations.append(f"MediaPipe extraction is slow ({fps:.1f} FPS). Reduce model_complexity or disable unused features (face/pose).")
            
            avg_time = results['mediapipe_extraction']['avg_time_ms']
            if avg_time > 50:
                recommendations.append(f"MediaPipe takes {avg_time:.2f}ms per frame. Consider using model_complexity=0 (Lite) for faster processing.")
        
        # Model inference recommendations
        if results['model_inference']['success']:
            fps = results['model_inference']['estimated_fps']
            if fps < 50:
                recommendations.append(f"Model inference is slow ({fps:.1f} FPS). Consider reducing model size or using mixed precision.")
            
            if results['model_inference']['device'] == 'cpu' and results['system']['cuda_available']:
                recommendations.append("Model running on CPU but CUDA is available. Move model to GPU for faster inference.")
        
        # Full pipeline recommendations
        if results['full_pipeline']['success']:
            fps = results['full_pipeline']['estimated_fps']
            if fps < 15:
                recommendations.append(f"Full pipeline is slow ({fps:.1f} FPS). This is below real-time requirements.")
            
            cpu_usage = results['full_pipeline']['avg_cpu_percent']
            if cpu_usage > 80:
                recommendations.append(f"High CPU usage ({cpu_usage:.1f}%). Consider optimizing feature extraction or using GPU acceleration.")
            
            memory_mb = results['full_pipeline']['max_memory_mb']
            if memory_mb > 1000:
                recommendations.append(f"High memory usage ({memory_mb:.1f}MB). Consider reducing batch size or model size.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major bottlenecks detected.")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_path: str = "profile_results.json"):
        """Save profiling results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main profiling function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile inference pipeline")
    parser.add_argument("--checkpoint", type=str, default="ml/models/checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="profile_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    profiler = InferenceProfiler()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Using mock profiling without actual model...")
        results = profiler.run_full_profile(str(checkpoint_path))
    else:
        results = profiler.run_full_profile(str(checkpoint_path))
    
    profiler.save_results(results, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    
    print(f"\nSystem Info:")
    print(f"  CPU Cores: {results['system']['cpu_count']}")
    print(f"  Total Memory: {results['system']['memory_total_gb']:.2f} GB")
    print(f"  CUDA Available: {results['system']['cuda_available']}")
    
    if results['model_loading']['success']:
        print(f"\nModel Loading:")
        print(f"  Load Time: {results['model_loading']['load_time_seconds']:.3f}s")
        print(f"  Model Size: {results['model_loading']['model_size_mb']:.2f}MB")
        print(f"  Parameters: {results['model_loading']['num_parameters']:,}")
    
    if results['webcam_capture']['success']:
        print(f"\nWebcam Capture:")
        print(f"  Avg Latency: {results['webcam_capture']['avg_latency_ms']:.2f}ms")
        print(f"  Estimated FPS: {results['webcam_capture']['estimated_fps']:.1f}")
    
    if results['mediapipe_extraction']['success']:
        print(f"\nMediaPipe Extraction:")
        print(f"  Avg Time: {results['mediapipe_extraction']['avg_time_ms']:.2f}ms")
        print(f"  Estimated FPS: {results['mediapipe_extraction']['estimated_fps']:.1f}")
    
    if results['model_inference']['success']:
        print(f"\nModel Inference:")
        print(f"  Avg Time: {results['model_inference']['avg_time_ms']:.2f}ms")
        print(f"  P95 Time: {results['model_inference']['p95_ms']:.2f}ms")
        print(f"  Estimated FPS: {results['model_inference']['estimated_fps']:.1f}")
        print(f"  Device: {results['model_inference']['device']}")
    
    if results['full_pipeline']['success']:
        print(f"\nFull Pipeline:")
        print(f"  Avg Frame Time: {results['full_pipeline']['avg_frame_time_ms']:.2f}ms")
        print(f"  Estimated FPS: {results['full_pipeline']['estimated_fps']:.1f}")
        print(f"  Avg CPU: {results['full_pipeline']['avg_cpu_percent']:.1f}%")
        print(f"  Avg Memory: {results['full_pipeline']['avg_memory_mb']:.1f}MB")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
