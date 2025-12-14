import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = "data/landmarks"

X = []
y = []

# --------------------------------
# LOAD DATA (ORDER MATTERS)
# --------------------------------
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        data = pd.read_csv(path, header=None)

        X.extend(data.iloc[:, :-1].values)
        y.extend(data.iloc[:, -1].values)

X = np.array(X)
y = np.array(y)

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
    tf.keras.layers.Dense(128, activation="relu", input_shape=(63,)),
    tf.keras.layers.Dense(64, activation="relu"),
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

print("✅ Model trained and saved correctly")
