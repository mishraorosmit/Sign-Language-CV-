"""
Configuration File for AI Sign Language to Speech Project

This file contains all the settings and parameters used throughout the project.
By keeping all settings in one place, it's easier to modify behavior without
changing multiple files.

BEGINNER NOTE: You can adjust these values to customize the project behavior.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
# These define where different files are stored in the project

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory where collected training data (CSV files) will be saved
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "collected")

# Directory where trained models will be saved
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# HAND DETECTION SETTINGS (MediaPipe)
# ============================================================================
# These control how MediaPipe detects hands from the webcam

# Minimum confidence (0.0 to 1.0) for hand detection
# Higher = more strict, fewer false positives
# Lower = more lenient, may detect non-hands
MIN_DETECTION_CONFIDENCE = 0.7

# Minimum confidence for tracking the hand between frames
# Higher = more stable but may lose tracking
# Lower = tracks better but may be jittery
MIN_TRACKING_CONFIDENCE = 0.5

# Maximum number of hands to detect
# Set to 2 to detect both hands, system will use the dominant one
MAX_NUM_HANDS = 2

# Hand selection preference
# 'auto' = automatically use dominant hand (higher confidence)
# 'left' = prefer left hand
# 'right' = prefer right hand
HAND_PREFERENCE = 'auto'

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================
# These control how training data is collected

# All 26 letters of the alphabet (A-Z)
# Each letter will have its own CSV file with landmark data
SIGN_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Number of samples to collect per letter
# More samples = better model, but takes longer to collect
SAMPLES_PER_LETTER = 100

# Delay (in seconds) between capturing each sample
# This gives you time to change your hand position slightly
SAMPLE_DELAY = 0.1

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================
# These control how the neural network is trained

# Name of the trained model file
MODEL_NAME = "sign_language_model.h5"

# Full path to the model file
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Percentage of data used for testing (0.0 to 1.0)
# 0.2 means 20% for testing, 80% for training
TEST_SIZE = 0.2

# Number of times the model sees all training data
# More epochs = better learning, but risk of overfitting
EPOCHS = 50

# Number of samples processed before updating model weights
# Smaller = more updates but slower
# Larger = faster but less frequent updates
BATCH_SIZE = 32

# Random seed for reproducibility
# Using the same seed gives the same train/test split every time
RANDOM_STATE = 42

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================
# These control real-time gesture prediction

# Minimum confidence to accept a prediction (0.0 to 1.0)
# Higher = only very confident predictions shown
# Lower = more predictions but may be wrong
PREDICTION_CONFIDENCE_THRESHOLD = 0.75

# Number of consecutive frames with same prediction before accepting it
# This prevents flickering between different letters
# Higher = more stable but slower to respond
STABILITY_FRAMES = 5

# ============================================================================
# TEXT FORMATION SETTINGS
# ============================================================================
# These control how letters become words and sentences

# Time (in seconds) to wait before adding the same letter again
# This prevents "AAAA" when you hold one gesture
LETTER_REPEAT_DELAY = 1.5

# Time (in seconds) of no gesture to trigger a space
# If no hand detected for this long, a space is added
SPACE_TRIGGER_DELAY = 2.0

# Maximum length of the sentence buffer
# Prevents memory issues with very long sentences
MAX_SENTENCE_LENGTH = 500

# ============================================================================
# TEXT-TO-SPEECH SETTINGS
# ============================================================================
# These control the voice output

# Speech rate (words per minute)
# Lower = slower, easier to understand
# Higher = faster
SPEECH_RATE = 150

# Voice volume (0.0 to 1.0)
SPEECH_VOLUME = 1.0

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
# These control the webcam

# Camera index (usually 0 for default webcam)
# If you have multiple cameras, try 1, 2, etc.
CAMERA_INDEX = 0

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Frames per second for display
FPS = 30

# ============================================================================
# UI SETTINGS
# ============================================================================
# These control the display window

# Window name for OpenCV display
WINDOW_NAME = "AI Sign Language to Speech"

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)

# Hand-specific colors
COLOR_LEFT_HAND = (255, 0, 255)   # Magenta for left hand
COLOR_RIGHT_HAND = (0, 255, 255)  # Yellow for right hand
COLOR_DOMINANT_HAND = (0, 255, 0)  # Green for dominant hand

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
# These help with troubleshooting

# Show detailed logs
VERBOSE = True

# Show landmark coordinates on screen
SHOW_LANDMARKS = True

# Show FPS counter
SHOW_FPS = True
