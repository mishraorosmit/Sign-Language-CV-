import cv2
import mediapipe as mp
import csv
import os

# ===============================
# CHANGE THIS LETTER EVERY RUN
# ===============================
LETTER = "U"   # Change to B, C, D, ...

SAMPLES = 200
DATA_DIR = "data/landmarks"

os.makedirs(DATA_DIR, exist_ok=True)
SAVE_PATH = os.path.join(DATA_DIR, f"{LETTER}.csv")

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0
print(f"Collecting data for letter: {LETTER}")

with open(SAVE_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)

    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(LETTER)
                writer.writerow(row)
                count += 1

        cv2.putText(
            frame,
            f"{LETTER} samples: {count}/{SAMPLES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete")
