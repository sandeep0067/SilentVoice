# SilentVoice: AI-Powered ASL Recognition System
## Exact Presentation Content (8 Slides)

---

## Slide 1: Title Slide

SilentVoice

Real-Time American Sign Language Recognition System

Hands-Free Communication Using AI/ML

Presented by: [Your Name]

AI/ML Course

[Current Date]

---

## Slide 2: Problem & Solution

70 million deaf and hard-of-hearing individuals face daily communication barriers. Most assistive technology requires keyboard or mouse input, which excludes sign language users. Existing ASL systems are slow, expensive, or require specialized hardware.

SilentVoice is a real-time, hands-free ASL recognition system that converts sign language to text and speech instantly using only a standard webcam. It requires zero keyboard input and is completely gesture-controlled. The system provides intelligent word suggestions for faster communication and supports natural two-handed interaction.

This makes technology truly accessible for the deaf community by removing the keyboard barrier and enabling natural sign language input.

---

## Slide 3: System Architecture

The system consists of three main layers. The User Interface Layer provides a real-time 1280x720 HD video feed with live prediction display, word suggestions with gesture selection, and text-to-speech audio output.

The AI/ML Processing Layer uses MediaPipe Hands for 21-point hand landmark extraction, creating a 63-dimensional feature vector from the landmarks. An MLP classifier handles 29 classes covering all letters plus space and delete commands. Temporal smoothing uses a 7-frame window with 5-frame stability requirements. The word suggestion engine combines prefix matching, context analysis, and user history.

The Output Layer displays text in a sentence buffer with word completion, provides real-time speech synthesis using pyttsx3, and shows visual progress indicators with color-coded feedback.

The key innovation is the two-hand gesture system where the primary hand signs ASL letters while the secondary hand controls selection with 1-3 fingers, clear with 4 fingers, and speak with 5 fingers. A 15-frame hold-to-confirm mechanism prevents accidental selections.

---

## Slide 4: Technical Implementation

The machine learning model is a Multi-Layer Perceptron with 63 input neurons representing hand landmark coordinates. The architecture includes three hidden layers with 128, 64, and 32 neurons respectively, using ReLU activation and dropout regularization. The output layer has 29 neurons with softmax activation to classify letters and commands.

The model was trained using cross-entropy loss with the Adam optimizer at a learning rate of 0.001. Training used a batch size of 32 over 50 epochs with early stopping, achieving 92% accuracy on the test set.

Three advanced features improve performance. Temporal smoothing uses a sliding window of 7 predictions and requires 5 consecutive stable predictions to confirm, reducing false positives by 80%. Similar gesture detection performs geometric analysis for confused letter pairs like M/N, R/V, and U/V using finger distance and angle calculations with confidence calibration. The word suggestion engine uses prefix matching, context-aware predictions based on common word pairs, and learns from user selection patterns.

---

## Slide 5: Two-Hand Gesture System

The two-hand gesture system enables completely hands-free operation. The primary hand signs ASL letters continuously, with suggestions appearing based on the partial word being built. This provides a natural signing experience.

The secondary hand controls the system through five distinct gestures. Holding 1 finger selects the first suggestion, 2 fingers selects the second, and 3 fingers selects the third. Holding 4 fingers clears the entire sentence, and holding 5 fingers speaks the full sentence.

The hold-to-confirm mechanism shows visual progress with a green circle filling during the hold. The duration is 15 frames or approximately 0.5 seconds, followed by a 10-frame cooldown after confirmation. Color-coded feedback provides immediate confirmation: green for selection, red for clear, and blue for speak.

This system provides zero keyboard dependency, natural two-handed interaction, persistent suggestions that don't disappear when the signing hand moves, and immediate audio and visual feedback.

---

## Slide 6: Real-Time Performance & UI

The system achieves 25 to 30 frames per second with latency under 50 milliseconds from camera capture to prediction. Model inference takes less than 5 milliseconds per frame, and text-to-speech latency is under 100 milliseconds. Accuracy metrics include 92% for letter recognition, 95% for gesture selection, and 85% for word completion with suggestions.

Resource usage is efficient with 30 to 40% CPU utilization on a single core, approximately 500MB of memory, no GPU requirement, and total storage of about 10MB.

The professional user interface features a header with SilentVoice branding, current prediction display with confidence percentage, a stability indicator bar, and speaking status indicator. The main display shows a full-screen webcam feed with hand landmark overlay and real-time prediction overlay. The footer contains the sentence buffer with word count, numbered suggestion boxes, gesture hints, and keyboard shortcuts. The interface automatically scales to resolution and supports fullscreen toggle.

---

## Slide 7: Demo & Use Cases

A live demo demonstrates signing the phrase "Hello, how are you?" The user signs H-E-L-L-O with the primary hand, and the system suggests hello, help, and here. Holding 1 finger with the secondary hand selects hello, which triggers auto-spacing and auto-speaking. The user then signs H-O-W and selects how, signs A-R-E and selects are, signs Y-O-U and selects you, then holds 5 fingers to speak the full sentence. The complete sentence is spoken in approximately 15 seconds.

Quick commands include holding 4 fingers to clear the sentence with a red flash for instant reset, holding 5 fingers to speak the sentence with blue flash and TTS output, and pressing the F key to toggle fullscreen.

Real-world applications include daily communication for conversations with non-signers, education for classroom participation, workplace meetings and presentations, emergency communication with emergency services, and social media for real-time caption generation.

---

## Slide 8: Conclusion & Future Work

SilentVoice achieves real-time ASL recognition with 92% accuracy, a two-hand gesture interface that is completely hands-free, intelligent word suggestions using a three-tier strategy, auto-speak text-to-speech integration, and a production-ready application with configuration support.

Key innovations include being the first system to use two-hand gesture control for ASL, a hold-to-confirm mechanism that prevents accidental selections, context-aware word suggestions that improve communication speed, and a responsive user interface with professional design.

Short-term future enhancements include expanded vocabulary with phrases and domain-specific terms, adjustable gesture sensitivity for different users, and high-contrast mode for better visibility. Long-term goals include true ASL grammar support beyond fingerspelling, international sign language support, mobile deployment for Android and iOS, multi-person detection for conversations, and facial expression integration.

SilentVoice demonstrates practical application of AI and machine learning for social good, shows how computer vision can improve accessibility, highlights the importance of user-centered design, and provides an open source contribution to assistive technology.

Questions?

