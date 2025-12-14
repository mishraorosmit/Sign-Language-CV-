import sys
import os

print("-" * 50)
print("ENVIRONMENT VERIFICATION")
print("-" * 50)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe Version: {mp.__version__}")
    print(f"   Location: {os.path.dirname(mp.__file__)}")
except ImportError as e:
    print(f"❌ MediaPipe NOT found: {e}")

try:
    import cv2
    print(f"✅ OpenCV Version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV NOT found: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow Version: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow NOT found: {e}")

print("-" * 50)
