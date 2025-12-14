import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import extract_geometric_features, augment_landmarks

DATA_DIR = "data/landmarks"

X = []
y = []

# --------------------------------
# LOAD DATA (ORDER MATTERS)
# --------------------------------
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        try:
            data = pd.read_csv(path, header=None)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Warning: Skipping empty file {file}")
            continue

        X.extend(data.iloc[:, :-1].values)
        y.extend(data.iloc[:, -1].values)

X = np.array(X)
y = np.array(y)

# --------------------------------
# CONVERT TO DISTANCE/ANGLE FEATURES + AUGMENTATION
# --------------------------------
print("Augmenting data...")
X_features = []
y_augmented = []

for i in range(len(X)):
    # 1. Generate augmented versions of landmarks (Original + Rotated + Scaled + Noisy)
    augmented_versions = augment_landmarks(X[i])
    
    # 2. Add EACH version to training data (Raw 63 landmarks)
    for aug_landmarks in augmented_versions:
        # We are NOT extracting geometric features anymore to avoid shape mismatch.
        # We use the raw augmented landmarks (Shape: 63)
        X_features.append(aug_landmarks)
        y_augmented.append(y[i]) # Same label for all variations

X = np.array(X_features)
y = np.array(y_augmented)

print(f"New augmented dataset shape: {X.shape}")

print("Raw labels:", np.unique(y))

# --------------------------------
# ENCODE LABELS CORRECTLY
# --------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Encoded labels:", np.unique(y))
print("Number of classes:", len(np.unique(y)))

# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------
# MODEL (OUTPUT MUST = CLASSES)
# --------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)), # Explicitly expecting 63 features now
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(32, activation="relu"),
    
    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

# --------------------------------
# SAVE MODEL (OVERWRITE OLD ONE)
# --------------------------------
os.makedirs("model", exist_ok=True)
model.save("model/gesture_model.h5")
np.save("model/gesture_model_labels.npy", encoder.classes_)

print("✅ Model trained and saved correctly")
