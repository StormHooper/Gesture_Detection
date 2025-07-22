import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pathlib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Output structure
data = []

# Signs you want to recognize
sign_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Webcam setup
cap = cv2.VideoCapture(0)

print("[Instructions]")
print("Press the key matching the label you want to save: A-Z or any custom sign.")
print("Press SPACE to record, 's' to skip frame, 'q' to quit and save.")

current_label = "A"
json_path = "dataset/asl_data.json"

# Load existing data if available
if pathlib.Path(json_path).exists():
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"[Loaded] Existing dataset with {len(data)} samples.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    h, w, _ = frame.shape
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    # Show label
    cv2.putText(frame, f"Current Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Data Collector (JSON)", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('0'):
        break

    elif key == ord('1'):
        continue

    elif chr(key).upper() in sign_list or chr(key).upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        current_label = chr(key).upper()

    elif key == ord(' '):  # Spacebar to save sample
        if landmark_list:
            sample = {
                "label": current_label,
                "landmarks": landmark_list
            }
            data.append(sample)
            print(f"[Saved] Sample for '{current_label}' — Total samples: {len(data)}")
        else:
            print("[Skip] No hand detected")

cap.release()
cv2.destroyAllWindows()

# Save to JSON
os.makedirs("dataset", exist_ok=True)
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"[Complete] Saved {len(data)} total samples to '{json_path}'")