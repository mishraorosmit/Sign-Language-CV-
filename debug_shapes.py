import numpy as np
import tensorflow as tf
from utils import extract_geometric_features
from config import MODEL_PATH

print("--- DEBUGGING SHAPES ---")

# 1. Test Feature Extractor
try:
    dummy_landmarks = np.random.rand(63)
    features = extract_geometric_features(dummy_landmarks)
    print(f"Dummy Landmarks Shape: {dummy_landmarks.shape}")
    print(f"Extracted Features Shape: {features.shape}")
    
    if features.shape[0] == 23:
        print("✓ Feature extraction produces 23 features (Correct)")
    else:
        print(f"❌ Feature extraction produces {features.shape[0]} features (Expected 23)")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")

# 2. Test Model Input Shape
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape
    print(f"Model Input Shape: {input_shape}")
    
    expected_dim = input_shape[1]
    print(f"Model expects input dimension: {expected_dim}")
    
    if expected_dim == 23:
        print("✓ Model expects 23 features (Correct)")
    elif expected_dim == 63:
        print("❌ Model expects 63 features (OLD MODEL?)")
    else:
        print(f"❌ Model expects {expected_dim} features (Unexpected)")

except Exception as e:
    print(f"❌ Could not load model: {e}")
