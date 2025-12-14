import sys
print(f"Python: {sys.version}")
try:
    import mediapipe as mp
    print(f"MediaPipe Version: {mp.__version__}")
    print("MediaPipe OK")
except Exception as e:
    print(f"MediaPipe Error: {e}")
