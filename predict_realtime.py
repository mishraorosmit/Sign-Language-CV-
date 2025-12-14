"""
Real-Time Sign Language Prediction

This script uses the trained model to recognize sign language gestures in real-time
from your webcam feed. It shows the predicted letter and confidence on screen.

BEGINNER NOTE: This is where everything comes together! The model you trained
will now recognize your hand gestures live.

HOW TO USE:
1. Make sure you've trained a model using train_model.py first
2. Run this script: python predict_realtime.py
3. Show sign language gestures to your webcam
4. Press 'q' to quit
"""

import cv2
import numpy as np
import time
from tensorflow import keras
from hand_detector import HandDetector
from config import (
    MODEL_PATH, CAMERA_INDEX, PREDICTION_CONFIDENCE_THRESHOLD,
    STABILITY_FRAMES
)


class RealtimePredictor:
    """
    A class to perform real-time sign language prediction.
    
    This loads the trained model and uses it to predict gestures from
    the webcam feed.
    """
    
    def __init__(self):
        """Initialize the real-time predictor."""
        print("\n" + "="*60)
        print("INITIALIZING REAL-TIME PREDICTOR")
        print("="*60)
        
        # Initialize hand detector
        self.detector = HandDetector()
        
        # Load the trained model
        print(f"Loading model from: {MODEL_PATH}")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"ERROR: Could not load model: {e}")
            print("Make sure you've trained a model using train_model.py first!")
            raise
        
        # Load label encoder
        label_encoder_path = MODEL_PATH.replace('.h5', '_labels.npy')
        try:
            self.labels = np.load(label_encoder_path)
            print(f"✓ Labels loaded: {self.labels}")
        except Exception as e:
            print(f"ERROR: Could not load labels: {e}")
            raise
        
        # Prediction stability tracking
        # We keep track of recent predictions to avoid flickering
        self.recent_predictions = []
        self.stable_prediction = None
        self.stable_confidence = 0.0
        
        print("="*60)
    
    def predict(self, landmarks_array):
        """
        Predict the sign language letter from landmarks.
        
        Args:
            landmarks_array: Normalized landmark array (63 values)
            
        Returns:
            Tuple of (predicted_letter, confidence)
            Returns (None, 0.0) if prediction fails
        """
        if landmarks_array is None:
            return None, 0.0
        
        # Reshape for model input
        # Model expects shape (batch_size, features)
        # We have 1 sample, so shape is (1, 63)
        landmarks_reshaped = landmarks_array.reshape(1, -1)
        
        # Make prediction
        # This returns probabilities for each class
        predictions = self.model.predict(landmarks_reshaped, verbose=0)
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Convert class index to letter
        predicted_letter = self.labels[predicted_class_idx]
        
        return predicted_letter, confidence
    
    def update_stable_prediction(self, letter, confidence):
        """
        Update stable prediction using recent predictions.
        
        This prevents flickering by requiring multiple consecutive
        predictions of the same letter before accepting it.
        
        Args:
            letter: Predicted letter
            confidence: Prediction confidence
        """
        # Add to recent predictions
        self.recent_predictions.append((letter, confidence))
        
        # Keep only last N predictions
        if len(self.recent_predictions) > STABILITY_FRAMES:
            self.recent_predictions.pop(0)
        
        # Check if we have enough predictions
        if len(self.recent_predictions) < STABILITY_FRAMES:
            return
        
        # Check if all recent predictions are the same
        recent_letters = [pred[0] for pred in self.recent_predictions]
        
        if len(set(recent_letters)) == 1:  # All same
            # All predictions agree - this is stable
            self.stable_prediction = recent_letters[0]
            # Use average confidence
            self.stable_confidence = np.mean([pred[1] for pred in self.recent_predictions])
    
    def run(self):
        """
        Run the real-time prediction loop.
        
        This opens the webcam and continuously predicts gestures.
        NOW WITH TWO-HAND SUPPORT!
        """
        print("\n" + "="*60)
        print("STARTING REAL-TIME PREDICTION - TWO HAND SUPPORT")
        print("="*60)
        print("Show sign language gestures with EITHER hand")
        print("The system will automatically use the dominant hand")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        print("✓ Webcam opened successfully")
        
        # For FPS calculation
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Could not read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect both hands
            hands_info, annotated_frame = self.detector.detect_both_hands(frame)
            
            # Get dominant hand
            dominant_hand = self.detector.get_dominant_hand(hands_info)
            
            # Prepare display
            display_frame = annotated_frame.copy()
            
            # If dominant hand detected, make prediction
            if dominant_hand is not None:
                # Get normalized landmarks from dominant hand
                landmarks_array = self.detector.get_normalized_landmarks(
                    dominant_hand['landmarks']
                )
                
                # Predict
                predicted_letter, confidence = self.predict(landmarks_array)
                
                # Update stable prediction
                if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                    self.update_stable_prediction(predicted_letter, confidence)
                else:
                    # Low confidence - reset
                    self.recent_predictions = []
                
                # Display prediction info
                self._draw_prediction_info(
                    display_frame,
                    predicted_letter,
                    confidence,
                    self.stable_prediction,
                    self.stable_confidence,
                    hands_info,
                    dominant_hand
                )
            else:
                # No hand detected
                self.recent_predictions = []
                self.stable_prediction = None
                
                # Show hands count
                if hands_info:
                    cv2.putText(
                        display_frame,
                        f"{len(hands_info)} HAND(S) DETECTED - No dominant hand",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "NO HANDS DETECTED",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow("Real-Time Sign Language Prediction - Two Hands", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        
        print("\n✓ Prediction stopped")
    
    def _draw_prediction_info(self, frame, letter, confidence, stable_letter, stable_confidence, hands_info=None, dominant_hand=None):
        """
        Draw prediction information on the frame.
        
        Args:
            frame: Frame to draw on
            letter: Current predicted letter
            confidence: Current confidence
            stable_letter: Stable prediction
            stable_confidence: Stable confidence
            hands_info: List of detected hands (optional)
            dominant_hand: The dominant hand being used (optional)
        """
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Show hands info
        if hands_info:
            hands_text = f"Hands: {len(hands_info)}"
            if dominant_hand:
                hands_text += f" | Using: {dominant_hand['handedness']} ({dominant_hand['confidence']*100:.1f}%)"
            
            cv2.putText(
                frame,
                hands_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        
        # Current prediction
        color = (0, 255, 0) if confidence >= PREDICTION_CONFIDENCE_THRESHOLD else (0, 165, 255)
        
        cv2.putText(
            frame,
            f"Current: {letter} ({confidence*100:.1f}%)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        
        # Stable prediction (if available)
        if stable_letter is not None:
            cv2.putText(
                frame,
                f"Stable: {stable_letter} ({stable_confidence*100:.1f}%)",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
            
            # Large letter display
            cv2.putText(
                frame,
                stable_letter,
                (width - 100, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5
            )
        
        # Threshold indicator
        cv2.putText(
            frame,
            f"Threshold: {PREDICTION_CONFIDENCE_THRESHOLD*100:.0f}%",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


def main():
    """Main function to run real-time prediction."""
    try:
        predictor = RealtimePredictor()
        predictor.run()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure you've trained a model first (train_model.py)")
        print("2. Check that your webcam is working")
        print("3. Make sure no other program is using the webcam")


if __name__ == "__main__":
    main()
