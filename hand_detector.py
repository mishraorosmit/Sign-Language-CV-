"""
Hand Detection Module using MediaPipe

This module handles all hand detection functionality using Google's MediaPipe library.
MediaPipe is a FREE, open-source framework for building ML pipelines.

BEGINNER NOTE: This is the "eyes" of our system - it finds hands in webcam images.
"""

import cv2
import mediapipe as mp
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS
from utils import normalize_landmarks, draw_landmarks_on_image


class HandDetector:
    """
    A class that detects hands and extracts landmarks using MediaPipe.
    
    WHAT IT DOES:
    1. Opens your webcam
    2. Looks for hands in each frame
    3. Finds 21 key points (landmarks) on the hand
    4. Returns normalized coordinates we can use for training
    """
    
    def __init__(self):
        """
        Initialize the hand detector.
        
        This sets up MediaPipe with our configuration settings.
        """
        # Initialize MediaPipe Hands
        # mp.solutions.hands gives us the hand detection functionality
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create the Hands object with our settings
        # static_image_mode=False means we're processing video, not static images
        # This makes tracking faster between frames
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        print("✓ Hand detector initialized successfully")
    
    def detect_hand(self, image):
        """
        Detect hand in an image and return landmarks.
        
        Args:
            image: BGR image from OpenCV (numpy array)
            
        Returns:
            Tuple of (hand_landmarks, annotated_image)
            - hand_landmarks: MediaPipe landmarks object (or None if no hand found)
            - annotated_image: Image with landmarks drawn on it
        """
        # MediaPipe expects RGB images, but OpenCV uses BGR
        # So we need to convert the color format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        # This is where the magic happens!
        results = self.hands.process(image_rgb)
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # We only care about the first hand (index 0)
            # Even if multiple hands are detected, we use just one
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on the image for visualization
            annotated_image = draw_landmarks_on_image(
                image, 
                hand_landmarks, 
                self.mp_hands, 
                self.mp_drawing
            )
            
            return hand_landmarks, annotated_image
        else:
            # No hand detected
            return None, image
    
    def detect_both_hands(self, image):
        """
        Detect both hands in an image and return detailed information.
        
        This is the NEW method that supports two-hand detection!
        
        Args:
            image: BGR image from OpenCV (numpy array)
            
        Returns:
            Tuple of (hands_info, annotated_image)
            - hands_info: List of dictionaries with hand information, or empty list
              Each dict contains: 'landmarks', 'handedness', 'confidence', 'label'
            - annotated_image: Image with landmarks drawn on both hands
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Start with the original image
        annotated_image = image.copy()
        hands_info = []
        
        # Check if any hands were detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Process each detected hand
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                # Get hand label (Left or Right) and confidence
                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score
                
                # Store hand information
                hand_info = {
                    'landmarks': hand_landmarks,
                    'handedness': hand_label,
                    'confidence': hand_confidence,
                    'label': f"{hand_label} ({hand_confidence*100:.1f}%)"
                }
                hands_info.append(hand_info)
                
                # Draw landmarks with hand-specific color
                from config import COLOR_LEFT_HAND, COLOR_RIGHT_HAND
                color = COLOR_LEFT_HAND if hand_label == 'Left' else COLOR_RIGHT_HAND
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=color, thickness=2)
                )
        
        return hands_info, annotated_image
    
    def get_dominant_hand(self, hands_info):
        """
        Get the dominant hand from detected hands based on preference and confidence.
        
        Args:
            hands_info: List of hand information dictionaries from detect_both_hands()
            
        Returns:
            The hand_info dict of the dominant hand, or None if no hands
        """
        if not hands_info:
            return None
        
        from config import HAND_PREFERENCE
        
        # If only one hand, return it
        if len(hands_info) == 1:
            return hands_info[0]
        
        # Multiple hands - apply preference
        if HAND_PREFERENCE == 'left':
            # Prefer left hand
            for hand in hands_info:
                if hand['handedness'] == 'Left':
                    return hand
            # If no left hand, return highest confidence
            return max(hands_info, key=lambda h: h['confidence'])
        
        elif HAND_PREFERENCE == 'right':
            # Prefer right hand
            for hand in hands_info:
                if hand['handedness'] == 'Right':
                    return hand
            # If no right hand, return highest confidence
            return max(hands_info, key=lambda h: h['confidence'])
        
        else:  # 'auto' or default
            # Return hand with highest confidence
            return max(hands_info, key=lambda h: h['confidence'])

    
    def get_normalized_landmarks(self, hand_landmarks):
        """
        Convert hand landmarks to normalized numpy array.
        
        Args:
            hand_landmarks: MediaPipe landmarks object
            
        Returns:
            Numpy array of shape (63,) with normalized coordinates
            Returns None if landmarks are invalid
        """
        return normalize_landmarks(hand_landmarks)
    
    def close(self):
        """
        Clean up resources.
        
        IMPORTANT: Always call this when you're done to free up memory.
        """
        self.hands.close()
        print("✓ Hand detector closed")


def test_hand_detector():
    """
    Test function to verify hand detection is working.
    
    This opens your webcam and shows detected hand landmarks in real-time.
    NOW SUPPORTS TWO HANDS!
    Press 'q' to quit.
    
    BEGINNER NOTE: Run this to make sure your webcam and MediaPipe are working!
    """
    print("\n" + "="*60)
    print("HAND DETECTOR TEST - TWO HAND DETECTION")
    print("="*60)
    print("This will open your webcam and detect BOTH hands.")
    print("Left hand = Magenta, Right hand = Yellow")
    print("Dominant hand will be highlighted in green box")
    print("Press 'q' to quit.")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = HandDetector()
    
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not being used by another program.")
        return
    
    print("✓ Webcam opened successfully")
    print("Show your hands to the camera...")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Could not read frame from webcam")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect both hands
        hands_info, annotated_frame = detector.detect_both_hands(frame)
        
        # Get dominant hand
        dominant_hand = detector.get_dominant_hand(hands_info)
        
        # Add status text
        if hands_info:
            # Hands detected
            cv2.putText(
                annotated_frame,
                f"HANDS DETECTED: {len(hands_info)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show each hand's info
            y_offset = 70
            for i, hand in enumerate(hands_info):
                color = (0, 255, 0) if hand == dominant_hand else (255, 255, 255)
                marker = "★ " if hand == dominant_hand else "  "
                
                cv2.putText(
                    annotated_frame,
                    f"{marker}{hand['label']}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                y_offset += 35
            
            # Highlight dominant hand
            if dominant_hand:
                cv2.putText(
                    annotated_frame,
                    f"Dominant: {dominant_hand['handedness']}",
                    (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        else:
            # No hands detected
            cv2.putText(
                annotated_frame,
                "NO HANDS DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # Show instructions
        cv2.putText(
            annotated_frame,
            "Press 'q' to quit",
            (10, annotated_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Display the frame
        cv2.imshow("Hand Detector Test - Two Hands", annotated_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\n✓ Test completed successfully!")


# If this file is run directly (not imported), run the test
if __name__ == "__main__":
    test_hand_detector()
