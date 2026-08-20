"""
Enhanced Main Live Inference Loop for ASL Alphabet Recognition.

This script implements an improved inference pipeline with:
- Advanced feature extraction for better gesture discrimination
- Temporal smoothing to reduce prediction flickering
- Similar gesture detection and handling
- Confidence calibration
- Context-aware sequence analysis

Designed to better distinguish similar gestures like 'm' vs 'n', 'r' vs 'v', etc.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

from model import AlphabetMLP, AlphabetMLPConfig
from landmark_extractor import HandLandmarkExtractor
from word_builder import WordBuilder
from enhanced_tts_handler import EnhancedTTSHandler
from advanced_feature_extractor import AdvancedFeatureExtractor, SimilarGestureDetector
from temporal_smoother import TemporalSmoother, GestureSequenceAnalyzer, ConfidenceCalibrator
from word_suggester import WordSuggester


def load_trained_model(checkpoint_path: str, config_path: str, device: torch.device):
    """
    Load the trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pt
        config_path: Path to model_config.json
        device: torch device (cuda or cpu)
        
    Returns:
        Loaded model and label mapping
    """
    print("Loading trained model...")
    
    # Load model config
    config_path = Path(config_path)
    if config_path.exists():
        model_config = AlphabetMLPConfig.load_config(config_path)
    else:
        print(f"Config file not found, using defaults")
        model_config = AlphabetMLPConfig()
    
    # Create model
    model = AlphabetMLP(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load label mapping if available, otherwise use default
    if 'label_mapping' in checkpoint:
        label_mapping = checkpoint['label_mapping']
        print(f"Loaded label mapping from checkpoint")
    else:
        # Default classes: A-Z + 'del' + 'nothing' + 'space' (sorted alphabetically)
        default_labels = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'nothing', 'space'])
        label_mapping = {i: label for i, label in enumerate(default_labels)}
        print(f"Using default label mapping: {list(label_mapping.values())}")
    
    print(f"Model loaded successfully on {device}")
    return model, label_mapping


def draw_enhanced_prediction_on_frame(frame, predicted_label, confidence_score, hand_detected, 
                                     sentence_buffer="", stability_progress=0.0, is_speaking=False,
                                     word_stats=None, smoothing_info=None, similar_gesture_warning=False,
                                     word_suggestions=None):
    """
    Draw enhanced prediction results with additional information.
    
    Shows detailed metrics about the prediction quality and similar gesture handling.
    """
    annotated = frame.copy()
    
    # Enhanced header bar with system status
    cv2.rectangle(annotated, (0, 0), (frame.shape[1], 90), (25, 25, 30), -1)
    
    if hand_detected:
        # Enhanced letter display with confidence color coding
        conf_pct = int(confidence_score * 100)
        
        # Color based on confidence and stability
        if smoothing_info and smoothing_info.get('is_smoothed'):
            if smoothing_info.get('stability_score', 0) > 0.7:
                conf_color = (0, 220, 120)  # Green for stable predictions
            else:
                conf_color = (200, 180, 50)  # Yellow for less stable
        else:
            conf_color = (0, 200, 100) if confidence_score > 0.7 else (200, 180, 50)
        
        # Main prediction display
        cv2.putText(annotated, predicted_label, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, conf_color, 3)
        
        # Confidence percentage
        cv2.putText(annotated, f"{conf_pct}%", (120, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
        
        # Similar gesture warning
        if similar_gesture_warning:
            cv2.putText(annotated, "~", (180, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 50), 2)
        
        # Enhanced stability bar
        if stability_progress > 0:
            bar_width = 120
            bar_height = 5
            bar_x = 20
            bar_y = 60
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 55), -1)
            progress_width = int(bar_width * stability_progress)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), conf_color, -1)
        
        # Additional metrics (small)
        if smoothing_info:
            stability = smoothing_info.get('stability_score', 0)
            cv2.putText(annotated, f"S:{stability:.2f}", (150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    else:
        cv2.putText(annotated, "No hand", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    
    # Enhanced sentence display at bottom
    if sentence_buffer:
        cv2.rectangle(annotated, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (25, 25, 30), -1)
        
        # Display sentence (truncate if too long)
        display_text = sentence_buffer if len(sentence_buffer) <= 40 else sentence_buffer[-40:]
        cv2.putText(annotated, display_text, (20, frame.shape[0] - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        
        # Display word suggestions
        if word_suggestions:
            suggestion_text = "Suggestions: " + " | ".join([f"{w}({c:.0f})" for w, c in word_suggestions])
            cv2.putText(annotated, suggestion_text, (20, frame.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            
            # Display hint for selecting suggestions
            cv2.putText(annotated, "[1-3] Select suggestion", (frame.shape[1] - 200, frame.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 200), 1)
        
        # Enhanced controls hint
        controls = "[s] Speak  [c] Clear  [r] Reset  [q] Quit"
        cv2.putText(annotated, controls, (frame.shape[1] - 320, frame.shape[0] - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 200), 1)
        
        # Word count
        if word_stats:
            word_count = word_stats.get('word_count', 0)
            cv2.putText(annotated, f"Words: {word_count}", (20, frame.shape[0] - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 200), 1)
    
    # Enhanced speaking indicator
    if is_speaking:
        cv2.circle(annotated, (frame.shape[1] - 30, 35), 10, (0, 200, 100), -1)
        cv2.putText(annotated, "●", (frame.shape[1] - 38, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)
    
    # Similar gesture warning indicator
    if similar_gesture_warning:
        cv2.rectangle(annotated, (frame.shape[1] - 50, 60), (frame.shape[1] - 10, 80), (255, 150, 50), -1)
        cv2.putText(annotated, "!", (frame.shape[1] - 38, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated


def main():
    """Main enhanced live inference loop."""
    
    print("=" * 70)
    print("Enhanced ASL Alphabet Recognition - Live Inference")
    print("Advanced features for similar gesture discrimination")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    script_dir = Path(__file__).parent
    checkpoint_path = str(script_dir / 'models' / 'best_model.pt')
    config_path = str(script_dir / 'models' / 'model_config.json')
    
    # Load model
    model, label_mapping = load_trained_model(checkpoint_path, config_path, device)
    
    # Initialize landmark extractor
    print("Initializing MediaPipe Hands...")
    landmark_extractor = HandLandmarkExtractor()
    
    # Initialize advanced feature extractor
    print("Initializing Advanced Feature Extractor...")
    advanced_extractor = AdvancedFeatureExtractor()
    
    # Initialize similar gesture detector
    print("Initializing Similar Gesture Detector...")
    similar_gesture_detector = SimilarGestureDetector()
    
    # Initialize temporal smoother
    print("Initializing Temporal Smoother...")
    temporal_smoother = TemporalSmoother(window_size=7, confidence_threshold=0.6)
    
    # Initialize gesture sequence analyzer
    print("Initializing Gesture Sequence Analyzer...")
    sequence_analyzer = GestureSequenceAnalyzer(sequence_length=15)
    
    # Initialize confidence calibrator
    print("Initializing Confidence Calibrator...")
    confidence_calibrator = ConfidenceCalibrator()
    
    # Initialize word builder with enhanced settings
    stability_frames = 5  # Balanced for accuracy and responsiveness
    confidence_threshold = 0.65  # Slightly higher for better quality
    print(f"Initializing WordBuilder (stability_frames={stability_frames}, confidence_threshold={confidence_threshold})")
    word_builder = WordBuilder(stability_frames=stability_frames, confidence_threshold=confidence_threshold)
    
    # Initialize Enhanced TTS handler
    clear_after_speaking = False
    voice_index = 2
    speech_rate = 130
    auto_speak_on_space = False
    print(f"Initializing Enhanced TTS Handler (voice={voice_index}, rate={speech_rate})")
    tts_handler = EnhancedTTSHandler(
        clear_after_speaking=clear_after_speaking,
        voice_index=voice_index,
        speech_rate=speech_rate,
        volume=0.9
    )
    
    # Initialize Word Suggester for auto-completion
    print("Initializing Word Suggester...")
    word_suggester = WordSuggester()
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    print("Starting enhanced live inference...")
    print("Controls: 'q' to quit, 'c' to clear sentence, 's' to speak, 'r' to reset smoothing")
    print("=" * 70)
    
    # Performance tracking
    frame_count = 0
    fps = 0
    start_time = cv2.getTickCount()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / ((end_time - start_time) / frame_count)
                print(f"FPS: {fps:.1f}")
            
            # Extract landmarks
            landmarks, results = landmark_extractor.extract_landmarks(frame)
            
            # Initialize prediction variables
            predicted_label = ""
            confidence_score = 0.0
            hand_detected = landmarks is not None
            sentence_buffer = word_builder.get_sentence()
            smoothing_info = None
            similar_gesture_warning = False
            word_suggestions = []
            
            if hand_detected:
                # Convert landmarks to tensor and add batch dimension
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    logits = model(landmarks_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    confidence, predicted = torch.max(probabilities, dim=1)
                
                # Get prediction
                predicted_class = predicted.item()
                raw_confidence = confidence.item()
                predicted_label = label_mapping.get(predicted_class, f"Class_{predicted_class}")
                
                # Get top-k predictions for similar gesture detection
                top_k_values, top_k_indices = torch.topk(probabilities, k=5, dim=1)
                top_k_predictions = [
                    (label_mapping.get(idx.item(), f"Class_{idx.item()}"), prob.item())
                    for idx, prob in zip(top_k_indices[0], top_k_values[0])
                ]
                
                # Calibrate confidence
                calibrated_confidence = confidence_calibrator.calibrate_confidence(
                    predicted_label, raw_confidence
                )
                
                # Apply temporal smoothing
                smoothing_result = temporal_smoother.add_prediction(
                    predicted_label, calibrated_confidence, landmarks
                )
                smoothing_info = {
                    'is_smoothed': smoothing_result['is_smoothed'],
                    'stability_score': smoothing_result.get('stability_score', 0.0)
                }
                
                # Use smoothed prediction if available
                if smoothing_result['is_smoothed']:
                    final_label = smoothing_result['class']
                    final_confidence = smoothing_result['confidence']
                else:
                    final_label = predicted_label
                    final_confidence = calibrated_confidence
                
                # Check for similar gesture confusion
                if len(top_k_predictions) >= 2:
                    refined_label, refined_conf = similar_gesture_detector.analyze_similar_gestures(
                        landmarks, final_label, final_confidence, top_k_predictions
                    )
                    
                    # Check if refinement changed the prediction
                    if refined_label != final_label:
                        similar_gesture_warning = True
                        final_label = refined_label
                        final_confidence = refined_conf
                
                # Apply contextual analysis
                contextual_label, contextual_conf = sequence_analyzer.get_contextual_suggestion(
                    final_label, final_confidence
                )
                
                # Add to sequence analyzer
                sequence_analyzer.add_gesture(contextual_label, contextual_conf)
                
                # Update word builder with final prediction
                sentence_buffer_before = sentence_buffer
                sentence_buffer, letter_confirmed = word_builder.update(
                    contextual_label, contextual_conf
                )
                
                # Get stability progress for visualization
                stability_progress = word_builder.get_stability_progress()
                
                # Get word suggestions based on current partial word
                current_word = word_builder.get_current_word()
                previous_word = sentence_buffer.split()[-1] if sentence_buffer.split() else ""
                word_suggestions = word_suggester.get_suggestions(current_word, previous_word, max_suggestions=3)
                
                # Update display variables
                predicted_label = contextual_label
                confidence_score = contextual_conf
                
                # Draw landmarks on frame
                frame = landmark_extractor.draw_landmarks(frame, results)
            else:
                # No hand detected
                sentence_buffer = word_builder.get_sentence()
                stability_progress = 0.0
                
                # Periodically print system status
                if frame_count % 60 == 0:
                    metrics = temporal_smoother.get_stability_metrics()
                    seq_stats = sequence_analyzer.get_sequence_stats()
                    print(f"System Status - Smoothing Stability: {metrics['stability_score']:.2f}, "
                          f"Sequence Unique: {seq_stats['unique_gestures']}")
            
            # Draw prediction on frame
            is_speaking = tts_handler.is_speaking()
            word_stats = word_builder.get_stats()
            annotated_frame = draw_enhanced_prediction_on_frame(
                frame, predicted_label, confidence_score, hand_detected, 
                sentence_buffer, stability_progress, is_speaking, word_stats,
                smoothing_info, similar_gesture_warning, word_suggestions
            )
            
            # Display frame
            cv2.imshow('Enhanced ASL Recognition', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            elif key == ord('c'):
                print("Clearing sentence")
                word_builder.clear()
            elif key == ord('r'):
                print("Resetting smoothing buffers")
                temporal_smoother.reset()
                sequence_analyzer.reset()
            elif key == ord('s'):
                # Speak the current sentence
                current_sentence = word_builder.get_sentence().strip()
                if current_sentence:
                    print(f"Manual speaking: '{current_sentence}'")
                    tts_handler.speak(current_sentence)
                    if tts_handler.clear_after_speaking:
                        print("Auto-clearing sentence after speaking")
                        word_builder.clear()
                else:
                    print("Sentence is empty, nothing to speak")
            elif key in [ord('1'), ord('2'), ord('3')] and word_suggestions:
                # Select a word suggestion
                suggestion_idx = key - ord('1')
                if suggestion_idx < len(word_suggestions):
                    selected_word, confidence = word_suggestions[suggestion_idx]
                    current_word = word_builder.get_current_word()
                    
                    # Remove current partial word
                    for _ in range(len(current_word)):
                        word_builder.update('del', 1.0)
                    
                    # Add selected word
                    for char in selected_word:
                        word_builder.update(char, 1.0)
                    
                    # Add space
                    word_builder.update('space', 1.0)
                    
                    # Record selection in history
                    word_suggester.select_suggestion(selected_word)
                    
                    print(f"Selected suggestion: {selected_word}")
                else:
                    print(f"Invalid suggestion index: {suggestion_idx + 1}")
            
            # Print detailed stats periodically
            if frame_count % 120 == 0 and hand_detected:
                print(f"\n=== Prediction Stats (Frame {frame_count}) ===")
                print(f"Raw: {predicted_label} ({raw_confidence:.3f})")
                print(f"Calibrated: {calibrated_confidence:.3f}")
                if smoothing_info:
                    print(f"Smoothed: {smoothing_result['class']} ({smoothing_result['confidence']:.3f})")
                    print(f"Stability: {smoothing_result['stability_score']:.2f}")
                if similar_gesture_warning:
                    print("⚠ Similar gesture handling applied")
                print(f"Final: {contextual_label} ({contextual_conf:.3f})")
                print(f"Top-3 predictions: {top_k_predictions[:3]}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        landmark_extractor.close()
        tts_handler.stop()
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        metrics = temporal_smoother.get_stability_metrics()
        print(f"Final smoothing stability: {metrics['stability_score']:.2f}")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        
        seq_stats = sequence_analyzer.get_sequence_stats()
        print(f"Unique gestures in sequence: {seq_stats['unique_gestures']}")
        
        print("Done!")


if __name__ == '__main__':
    main()