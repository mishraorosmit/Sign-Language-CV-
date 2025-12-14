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
from utils import extract_geometric_features


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
        # Model expects shape (1, 63) - Raw Augumented Landmarks
        
        # Reshape to (1, N)
        feature_vector = landmarks_array.reshape(1, -1)
        
        # Make prediction
        # This returns probabilities for each class
        predictions = self.model.predict(feature_vector, verbose=0)
        
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
        Draw prediction information with a sophisticated, modern UI.
        Theme: Cyan & Dark Grey
        """
        height, width = frame.shape[:2]
        
        # Colors (BGR)
        CYAN = (255, 255, 0)
        DARK_GREY = (30, 30, 30)
        GLASS_GREY = (20, 20, 20)
        WHITE = (255, 255, 255)
        
        # ---------------------------------------------------------
        # 1. Glassmorphism Sidebar
        # ---------------------------------------------------------
        sidebar_width = 320
        overlay = frame.copy()
        
        # Sidebar background
        cv2.rectangle(overlay, (0, 0), (sidebar_width, height), GLASS_GREY, -1)
        
        # Apply transparency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Sidebar accent line
        cv2.line(frame, (sidebar_width, 0), (sidebar_width, height), CYAN, 2)

        # ---------------------------------------------------------
        # 2. Header
        # ---------------------------------------------------------
        cv2.rectangle(frame, (0, 0), (sidebar_width, 60), (40, 40, 40), -1)
        # Gradient-like effect text? Just crisp Cyan text
        cv2.putText(frame, "AI GESTURES", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, CYAN, 1, cv2.LINE_AA)
        
        # ---------------------------------------------------------
        # 3. Connection / Hand Status
        # ---------------------------------------------------------
        y_cursor = 100
        
        status_color = CYAN if hands_info else (0, 0, 200) # Cyan or Dark Red
        status_text = "SYSTEM ACTIVE" if hands_info else "SEARCHING..."
        
        # Status Badge
        cv2.rectangle(frame, (20, y_cursor - 20), (180, y_cursor + 10), (50, 50, 50), -1)
        cv2.circle(frame, (35, y_cursor - 5), 5, status_color, -1)
        cv2.putText(frame, status_text, (50, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        
        y_cursor += 40
        if hands_info:
            hand_count_text = f"TRACKING: {len(hands_info)} HAND(S)"
            cv2.putText(frame, hand_count_text, (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y_cursor += 25
            if dominant_hand:
                dom_text = f"DOMAIN: {dominant_hand['handedness'].upper()}"
                cv2.putText(frame, dom_text, (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        # ---------------------------------------------------------
        # 4. Prediction Card (Hero)
        # ---------------------------------------------------------
        y_cursor = 220
        cv2.putText(frame, "ANALYSIS", (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        y_cursor += 15
        
        # Card Background
        cv2.rectangle(frame, (20, y_cursor), (sidebar_width - 20, y_cursor + 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, y_cursor), (sidebar_width - 20, y_cursor + 100), CYAN, 1) # Border
        
        display_letter = letter if letter else "-"
        text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)[0]
        text_x = 20 + (sidebar_width - 40 - text_size[0]) // 2
        text_y = y_cursor + 100 - (100 - text_size[1]) // 2
        
        cv2.putText(frame, display_letter, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 2.5, WHITE, 3, cv2.LINE_AA)
        
        # ---------------------------------------------------------
        # 5. Confidence Meter
        # ---------------------------------------------------------
        y_cursor += 130
        cv2.putText(frame, f"CONFIDENCE: {confidence*100:.0f}%", (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y_cursor += 10
        
        # Bar
        bar_width = sidebar_width - 40
        bar_height = 6
        cv2.rectangle(frame, (20, y_cursor), (20 + bar_width, y_cursor + bar_height), (50, 50, 50), -1)
        
        fill_width = int(bar_width * confidence)
        cv2.rectangle(frame, (20, y_cursor), (20 + fill_width, y_cursor + bar_height), CYAN, -1)
        
        # ---------------------------------------------------------
        # 6. Locked Output
        # ---------------------------------------------------------
        y_cursor += 60
        cv2.putText(frame, "LOCKED SIGNAL", (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        y_cursor += 40
        
        stable_display = stable_letter if stable_letter else "..."
        cv2.putText(frame, stable_display, (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 2, WHITE, 3, cv2.LINE_AA)
        
        # ---------------------------------------------------------
        # 7. EXCLUSIVE WATERMARK (Bottom Right)
        # ---------------------------------------------------------
        # "OroNab pvt ltd"
        watermark_text = "OroNab pvt ltd"
        (wm_w, wm_h), _ = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
        
        wm_x = width - wm_w - 20
        wm_y = height - 20
        
        # Subtle shadow for readability
        cv2.putText(frame, watermark_text, (wm_x + 1, wm_y + 1), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        # Main text
        cv2.putText(frame, watermark_text, (wm_x, wm_y), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # ---------------------------------------------------------
        # Main View Center Popup
        # ---------------------------------------------------------
        if stable_letter:
            main_center_x = sidebar_width + (width - sidebar_width) // 2
            main_y = height - 80
            
            label = f"DETECTED: {stable_letter}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            box_tl = (main_center_x - w//2 - 20, main_y - h - 20)
            box_br = (main_center_x + w//2 + 20, main_y + 10)
            
            sub_overlay = frame.copy()
            cv2.rectangle(sub_overlay, box_tl, box_br, (0, 0, 0), -1)
            cv2.addWeighted(sub_overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.rectangle(frame, box_tl, box_br, CYAN, 2)
            cv2.putText(frame, label, (main_center_x - w//2, main_y), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)


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
