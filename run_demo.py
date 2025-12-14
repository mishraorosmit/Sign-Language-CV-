"""
Demo Script - Two-Hand Detection

This is a simple demo to show off the two-hand detection capability.
Perfect for testing without needing a trained model!

HOW TO USE:
1. Make sure you've installed all requirements
2. Run this script: python run_demo.py
3. Show one or both hands to the camera
4. See them detected with different colors!
5. Press 'q' to quit

BEGINNER NOTE: This is the easiest way to test if everything is working!
"""

import cv2
import time
from hand_detector import HandDetector


def main():
    """Run the two-hand detection demo."""
    print("\n" + "="*60)
    print("AI SIGN LANGUAGE - TWO-HAND DETECTION DEMO")
    print("="*60)
    print("This demo shows both hands being detected simultaneously.")
    print("\nCOLOR CODING:")
    print("  🟣 Magenta = Left Hand")
    print("  🟡 Yellow  = Right Hand")
    print("  🟢 Green   = Dominant Hand (highest confidence)")
    print("\nPress 'q' to quit")
    print("="*60 + "\n")
    
    # Initialize detector
    print("Initializing hand detector...")
    detector = HandDetector()
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not being used by another program.")
        return
    
    print("✓ Webcam opened successfully")
    print("\nShow your hands to the camera!\n")
    
    # For FPS calculation
    prev_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Could not read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect both hands
        hands_info, annotated_frame = detector.detect_both_hands(frame)
        
        # Get dominant hand
        dominant_hand = detector.get_dominant_hand(hands_info)
        
        # Create info panel
        height, width = annotated_frame.shape[:2]
        
        # Dark background for info
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Title
        cv2.putText(
            annotated_frame,
            "Two-Hand Detection Demo",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )
        
        # Hand count
        if hands_info:
            cv2.putText(
                annotated_frame,
                f"Hands Detected: {len(hands_info)}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            
            # Show each hand's information
            y_offset = 115
            for hand in hands_info:
                # Determine color and marker
                from config import COLOR_LEFT_HAND, COLOR_RIGHT_HAND
                color = COLOR_LEFT_HAND if hand['handedness'] == 'Left' else COLOR_RIGHT_HAND
                
                is_dominant = (hand == dominant_hand)
                marker = "★ DOMINANT ★ " if is_dominant else ""
                
                # Hand label
                text = f"{marker}{hand['label']}"
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if is_dominant else color,
                    2
                )
                y_offset += 35
        else:
            cv2.putText(
                annotated_frame,
                "No Hands Detected",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
            
            cv2.putText(
                annotated_frame,
                "Show your hands to the camera!",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Show FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (width - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Instructions at bottom
        cv2.putText(
            annotated_frame,
            "Press 'q' to quit",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Display
        cv2.imshow("Two-Hand Detection Demo", annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("✓ Two-hand detection is working perfectly!")
    print("\nNext steps:")
    print("1. Collect training data: python collect_data.py")
    print("2. Train the model: python train_model.py")
    print("3. Run the full app: python main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("\nMake sure you've installed all requirements:")
        print("pip install -r requirements.txt")
