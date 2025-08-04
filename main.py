import cv2
import mediapipe as mp
import os
import time as t
import subprocess
import joblib
from pathlib import Path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# --- macOS Audio Control via osascript ---
def volume_up(step=10):
    subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) + {step})"])
    print(f"[Action] Volume increased by {step}.")

def volume_down(step=10):
    subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) - {step})"])
    print(f"[Action] Volume decreased by {step}.")

def toggle_mute():
    subprocess.run(["osascript", "-e", "set volume output muted not (output muted of (get volume settings))"])
    print(f"[Action] Audio mute toggled.")

# --- Brightness placeholders for macOS ---
def brightness_up(step=10):
    print(f"[Action] Brightness would increase by {step}% (macOS implementation skipped).")

def brightness_down(step=10):
    print(f"[Action] Brightness would decrease by {step}% (macOS implementation skipped).")

# --- ASL Model Load ---
model_path = Path("model/face_knn_model.joblib")
encoder_path = Path("model/label_encoder.joblib")

if model_path.exists() and encoder_path.exists():
    knn_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("[Model] ASL recognition model loaded.")
else:
    knn_model = None
    label_encoder = None
    print("[Warning] No ASL model found. ASL recognition will be disabled.")

# Mediapipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- State ---
active_read = False
cooldown_secs = 3.0
last_event_time = 0
last_command_time = 0
command_cooldown_secs = 1.5
asl_active = False
last_asl_letter = None
last_printed = None

# --- Actions ---
def do_open_palm_action():
    print("[Action] Simulated: Turning lights ON.")

def do_pointing_action():
    print("[Action] Increasing volume.")
    volume_up()

def do_peace_action():
    print("[Action] Toggling mute.")
    toggle_mute()

def do_bird_action():
    print("[Action] Middle finger detected. No action.")

def do_phone_action():
    print("[Action] Lowering brightness.")
    brightness_down()

def do_thumbs_up_action():
    print("[Action] Increasing brightness.")
    brightness_up()

def do_other_bird_action():
    print("[Action] Lowering volume.")
    volume_down()

# --- Feature Extraction ---
def extract_hand_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten().reshape(1, -1)

# --- Sign Recognition ---
def recognize_sign(frame):
    if not knn_model or not label_encoder:
        return "Model not loaded"
    features = extract_hand_features(frame)
    if features is None:
        return "No hand"
    prediction = knn_model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    return label

# --- Main Loop ---
print("Press 'q' to quit.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    gesture = "Unknown"
    now = t.time()

    if results.multi_hand_landmarks:
        if now - last_command_time < command_cooldown_secs:
            continue

        for lm_set in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)
            lm = lm_set.landmark

            # --- Finger States ---
            tips = [8, 12, 16, 20]
            fingers = [(lm[p].y < lm[p-2].y) for p in tips]
            thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
            fingers.insert(0, thumb_up)

            # --- Activation ---
            if fingers == [False]*5 and not active_read:
                gesture = "Fist"
                active_read = True
                last_event_time = now
                print("Fist detected — command mode will start in "
                      f"{cooldown_secs:.0f}s…")
                break

            if active_read and now - last_event_time < cooldown_secs:
                break

            # --- Commands ---
            if active_read and not asl_active:
                if fingers == [True]*5:
                    gesture, active_read = "Open Palm", False
                    do_open_palm_action()
                elif fingers == [False, True, False, False, False]:
                    gesture, active_read = "Pointing", False
                    do_pointing_action()
                elif fingers == [False, True, True, False, False]:
                    gesture, active_read = "Peace", False
                    do_peace_action()
                elif fingers == [False, False, True, False, False]:
                    gesture, active_read = "Bird", False
                    do_bird_action()
                elif fingers == [True, False, False, False, True]:
                    gesture, active_read = "Phone", False
                    do_phone_action()
                elif fingers == [True, False, False, False, False]:
                    gesture, active_read = "Thumbs Up", False
                    do_thumbs_up_action()
                elif fingers == [False, False, False, False, True]:
                    gesture, active_read = "Other Bird", False
                    do_other_bird_action()
                elif fingers == [True, True, True, False, False]:
                    gesture, active_read = "OK", False
                    asl_active = True
                    print("[Mode] ASL detection activated.")

                last_command_time = now

            # --- ASL Recognition Mode ---
            elif asl_active:
                prediction_label = recognize_sign(frame)
                if prediction_label != last_asl_letter and prediction_label not in ("No hand", "Model not loaded"):
                    print(f"[ASL] Detected: {prediction_label}")
                    last_asl_letter = prediction_label

                cv2.putText(frame, f"ASL: {prediction_label}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    else:
        if asl_active:
            print("[Mode] ASL detection stopped — no hand detected.")
        asl_active = False

    if gesture != "Unknown" and gesture != last_printed:
        print(f"Gesture: {gesture}")
        last_printed = gesture

    cv2.imshow('Gesture Recognition (macOS)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
