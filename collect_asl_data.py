import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pathlib
import platform

# Check platform (info only, but helpful for debugging)
if platform.system() != "Windows":
    print("[Info] Non-Windows OS detected — macOS webcam & I/O mode active.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Output structure
data = []

# ASL Signs (can be customized)
sign_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[Error] Cannot access webcam. Ensure camera permissions are granted in System Settings.")

# File path
json_path = "dataset/asl_data.json"
os.makedirs("dataset", exist_ok=True)

# Load existing data
if pathlib.Path(json_path).exists():
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"[Loaded] Existing dataset with {len(data)} samples.")

# UI instructions
print("[Instructions]")
print("Press a key (A–Z) to change label.")
print("Press SPACE to save a sample.")
print("Press ';' to quit and save.\n")

current_label = "A"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[Error] Frame grab failed.")
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

    # Show current label on frame
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Data Collector (macOS)", frame)
    key = cv2.waitKey(10)

    if key == -1:
        continue

    if key == ord(';'):
        break
    elif key == ord(' '):  # Save sample
        if landmark_list:
            sample = {"label": current_label, "landmarks": landmark_list}
            data.append(sample)
            print(f"[Saved] '{current_label}' — Total: {len(data)}")
        else:
            print("[Skip] No hand detected")
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
        current_label = chr(key).upper()

cap.release()
cv2.destroyAllWindows()

# Save JSON data
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"[Complete] Saved {len(data)} total samples to '{json_path}'")
