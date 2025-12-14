import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import pyttsx3

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
MODEL_PATH = "model/gesture_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully")

# ==================================================
# CLASS LABELS (A–Z) — BUG FREE
# ==================================================
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# ==================================================
# TEXT TO SPEECH (OFFLINE)
# ==================================================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ==================================================
# MEDIAPIPE HAND SETUP
# ==================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==================================================
# TEXT BUILDING VARIABLES
# ==================================================
sentence = ""
current_letter = None
last_added_letter = None
letter_start_time = 0

LETTER_HOLD_TIME = 1.5      # seconds
CONFIDENCE_THRESHOLD = 0.85

# ==================================================
# OPEN WEBCAM
# ==================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not accessible")
    exit()

print("Press:")
print("  SPACE → add space")
print("  ENTER → speak sentence")
print("  Q → quit")

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    predicted_letter = ""
    confidence = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ------------------------------------------
            # EXTRACT LANDMARKS (63 VALUES)
            # ------------------------------------------
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, 63)

            # ------------------------------------------
            # PREDICT LETTER
            # ------------------------------------------
            prediction = model.predict(landmarks, verbose=0)
            confidence = float(np.max(prediction))
            predicted_letter = labels[int(np.argmax(prediction))]

            # ------------------------------------------
            # STABILITY CHECK (ANTI-FLICKER)
            # ------------------------------------------
            if confidence > CONFIDENCE_THRESHOLD:
                if predicted_letter != current_letter:
                    current_letter = predicted_letter
                    letter_start_time = time.time()

                elif time.time() - letter_start_time >= LETTER_HOLD_TIME:
                    if predicted_letter != last_added_letter:
                        sentence += predicted_letter
                        last_added_letter = predicted_letter
                        letter_start_time = time.time()

    else:
        current_letter = None

    # ==================================================
    # DISPLAY UI
    # ==================================================
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Letter: {predicted_letter}  Conf: {confidence:.2f}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Text: {sentence}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Sign Language to Text & Speech", frame)

    # ==================================================
    # KEY CONTROLS
    # ==================================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        sentence += " "
        last_added_letter = None

    elif key == 13:  # ENTER key
        if sentence.strip():
            speak(sentence)

# ==================================================
# CLEANUP
# ==================================================
cap.release()
cv2.destroyAllWindows()
