"""
Main Live Inference Loop for ASL Alphabet Recognition.

This script runs real-time ASL alphabet recognition using:
- Webcam input
- MediaPipe Hands for landmark extraction
- Trained AlphabetMLP model for classification
- OpenCV for display
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from model import AlphabetMLP, AlphabetMLPConfig
from landmark_extractor import HandLandmarkExtractor
from word_builder import WordBuilder
from tts_handler import TTSHandler


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
    
    # Set model to not track gradients and disable batch norm updates
    for param in model.parameters():
        param.requires_grad = False
    
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


def draw_prediction_on_frame(frame, predicted_label, confidence_score, hand_detected, sentence_buffer="", stability_progress=0.0, is_speaking=False):
    """
    Draw prediction results on frame.
    """
    annotated = frame.copy()
    
    # Header bar
    cv2.rectangle(annotated, (0, 0), (frame.shape[1], 70), (30, 30, 35), -1)
    
    if hand_detected:
        # Display predicted letter
        cv2.putText(annotated, predicted_label, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 100), 2)
        
        # Display confidence
        conf_pct = int(confidence_score * 100)
        conf_color = (0, 200, 100) if confidence_score > 0.7 else (200, 180, 50)
        cv2.putText(annotated, f"{conf_pct}%", (100, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
        
        # Stability bar
        if stability_progress > 0:
            bar_width = 100
            bar_height = 4
            bar_x = 20
            bar_y = 55
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 55), -1)
            progress_width = int(bar_width * stability_progress)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 200, 100), -1)
    else:
        cv2.putText(annotated, "No hand", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    
    # Sentence display at bottom
    if sentence_buffer:
        cv2.rectangle(annotated, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (30, 30, 35), -1)
        
        # Display sentence
        display_text = sentence_buffer if len(sentence_buffer) <= 40 else sentence_buffer[-40:]
        cv2.putText(annotated, display_text, (20, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        
        # Controls hint
        controls = "[s] Speak  [c] Clear  [q] Quit"
        cv2.putText(annotated, controls, (frame.shape[1] - 280, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 200), 1)
    
    # Speaking indicator
    if is_speaking:
        cv2.circle(annotated, (frame.shape[1] - 30, 35), 8, (0, 200, 100), -1)
    
    return annotated


def main():
    """Main live inference loop."""
    
    print("=" * 60)
    print("ASL Alphabet Recognition - Live Inference")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model paths
    checkpoint_path = '../best_model.pt'
    config_path = '../model_config.json'
    
    # Load model
    model, label_mapping = load_trained_model(checkpoint_path, config_path, device)
    
    # Initialize landmark extractor
    print("Initializing MediaPipe Hands...")
    landmark_extractor = HandLandmarkExtractor()
    
    # Initialize word builder
    word_builder = WordBuilder(stability_frames=4, confidence_threshold=0.65)
    print("Initializing WordBuilder")
    
    # Initialize TTS handler
    tts_handler = TTSHandler(clear_after_speaking=False)
    print("Initializing TTS Handler")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    print("Starting live inference...")
    print("Controls: 'q' to quit, 'c' to clear sentence, 's' to speak")
    print("=" * 60)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame")
                break
            
            # Extract landmarks
            landmarks, results = landmark_extractor.extract_landmarks(frame)
            
            # Initialize prediction variables
            predicted_label = ""
            confidence_score = 0.0
            hand_detected = landmarks is not None
            sentence_buffer = word_builder.get_sentence()
            stability_progress = 0.0
            
            if hand_detected:
                # Convert landmarks to tensor
                if isinstance(landmarks, list):
                    landmarks_array = np.array(landmarks[0], dtype=np.float32)
                else:
                    landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Ensure shape is (63,)
                if landmarks_array.shape != (63,):
                    landmarks_array = landmarks_array.flatten()[:63]
                
                landmarks_tensor = torch.FloatTensor(landmarks_array).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    logits = model(landmarks_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    confidence, predicted = torch.max(probabilities, dim=1)
                
                # Get prediction
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                predicted_label = label_mapping.get(predicted_class, f"Class_{predicted_class}")
                
                # Update word builder
                sentence_buffer, letter_confirmed = word_builder.update(predicted_label, confidence_score)
                stability_progress = word_builder.get_stability_progress()
                
                # Draw landmarks
                frame = landmark_extractor.draw_landmarks(frame, results)
            
            # Draw prediction on frame
            is_speaking = tts_handler.is_speaking()
            annotated_frame = draw_prediction_on_frame(frame, predicted_label, confidence_score, hand_detected, 
                                                      sentence_buffer, stability_progress, is_speaking)
            
            # Display frame
            cv2.imshow('ASL Alphabet Recognition', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            elif key == ord('c'):
                print("Clearing sentence")
                word_builder.clear()
            elif key == ord('s'):
                current_sentence = word_builder.get_sentence().strip()
                if current_sentence:
                    print(f"Speaking: '{current_sentence}'")
                    tts_handler.speak(current_sentence)
                else:
                    print("Sentence is empty, nothing to speak")
                
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
        print("Done!")


if __name__ == '__main__':
    main()
