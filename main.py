# -------------------------------------------------
# Raspberry Pi 4 Gesture Recognition (Optimized)
# -------------------------------------------------

# --- Hardware Check ---
import platform
from pathlib import Path

if platform.system() != "Linux":
    raise EnvironmentError("This script must be run on a Linux-based Raspberry Pi device.")

model_path = Path("/proc/device-tree/model")
if not model_path.exists() or "Raspberry Pi" not in model_path.read_text():
    raise EnvironmentError("This script is intended for Raspberry Pi hardware only.")

print("[System] Raspberry Pi detected. Continuing...")

# --- Suppress Known Warnings ---
import warnings
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated",
    category=UserWarning
)

# --- Imports ---
import cv2
import mediapipe as mp
import os
import time as t
import joblib

# --- Camera Setup (Pi-Optimized) ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 for Pi camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution = higher FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            # Target FPS

# Camera warm-up
t.sleep(2)
print("[Camera] Initialized at 640x480 @ 30FPS.")

# --- Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Load ASL Model ---
model_dir = Path("model")
asl_model = label_encoder = None
if (model_dir / "asl_knn_model.joblib").exists() and (model_dir / "label_encoder.joblib").exists():
    asl_model = joblib.load(model_dir / "asl_knn_model.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    print("[Model] ASL recognition model loaded.")
else:
    print("[Warning] No ASL model found. ASL recognition disabled.")

# --- State Variables ---
active_read = False
cooldown_secs = 3.0
last_event_time = 0
last_printed = None
command_cooldown_secs = 1.5
last_command_time = 0
asl_active = False
last_asl_letter = None
asl_start_time = 0

# --- System Actions ---
def brightness_up(step=10):
    try:
        os.system(f"xrandr --output $(xrandr | grep ' connected' | cut -f1 -d ' ') --brightness 1")
        print(f"[Action] Brightness increased")
    except Exception as e:
        print(f"[Error] Brightness control failed: {e}")

def brightness_down(step=10):
    try:
        os.system(f"xrandr --output $(xrandr | grep ' connected' | cut -f1 -d ' ') --brightness 0.5")
        print(f"[Action] Brightness decreased")
    except Exception as e:
        print(f"[Error] Brightness control failed: {e}")

def toggle_mute():
    os.system("amixer set Master toggle")
    print("[Action] Toggled mute")

def volume_up(step=10):
    os.system(f"amixer set Master {step}%+")
    print(f"[Action] Volume increased by {step}%")

def volume_down(step=10):
    os.system(f"amixer set Master {step}%-")
    print(f"[Action] Volume decreased by {step}%")

# --- Gesture Action Mappings ---
def do_open_palm_action(): print("[Action] Turning lights ON (simulated).")
def do_pointing_action():  print("[Action] Pointer — increasing volume."); volume_up()
def do_peace_action():     print("[Action] Peace — muting audio."); toggle_mute()
def do_bird_action():      print("[Action] Middle finger — rude! Logging it as a joke.")
def do_phone_action():     print("[Action] Phone — lowering brightness."); brightness_down()
def do_thumbs_up_action(): print("[Action] Thumbs up — upping brightness."); brightness_up()
def do_other_bird_action():print("[Action] Pinky — lowering volume."); volume_down()

# --- Main Loop ---
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
) as hands:
    print("Press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "Unknown"
        now = t.time()
        if now - last_command_time < command_cooldown_secs:
            continue

        if results.multi_hand_landmarks:
            for lm_set in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)
                lm = lm_set.landmark

                # ASL detection
                if asl_model and asl_active and (now - asl_start_time > 1.0):
                    asl_features = [v for pt in lm_set.landmark for v in (pt.x, pt.y, pt.z)]
                    prediction = asl_model.predict([asl_features])[0]
                    predicted_sign = label_encoder.inverse_transform([prediction])[0]
                    if predicted_sign != last_asl_letter:
                        print(f"[ASL] Detected: {predicted_sign}")
                        last_asl_letter = predicted_sign
                    cv2.putText(frame, f"ASL: {predicted_sign}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                tips = [8, 12, 16, 20]
                fingers = [(lm[p].y < lm[p - 2].y) for p in tips]
                thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
                fingers.insert(0, thumb_up)

                if fingers == [False]*5 and not active_read:
                    gesture = "Fist"
                    active_read, last_event_time = True, now
                    print("Fist detected — command mode in "
                          f"{cooldown_secs:.0f}s…")
                    break

                if active_read and now - last_event_time < cooldown_secs:
                    break

                if active_read and not asl_active:
                    if fingers == [True]*5:                       gesture, active_read = "Open Palm", False; do_open_palm_action()
                    elif fingers == [False, True, False, False, False]: gesture, active_read = "Pointing", False; do_pointing_action()
                    elif fingers == [False, True, True, False, False]: gesture, active_read = "Peace", False; do_peace_action()
                    elif fingers == [False, False, True, False, False]: gesture, active_read = "Bird", False; do_bird_action()
                    elif fingers == [True, False, False, False, True]:  gesture, active_read = "Phone", False; do_phone_action()
                    elif fingers == [True, False, False, False, False]: gesture, active_read = "Thumbs Up", False; do_thumbs_up_action()
                    elif fingers == [False, False, False, False, True]: gesture, active_read = "Other Bird", False; do_other_bird_action()
                    elif fingers == [True, True, True, False, False]:
                        gesture, active_read = "OK", False
                        asl_active, asl_start_time = True, now
                        print("[Mode] ASL detection activated.")
                    last_command_time = now
        else:
            if asl_active:
                print("[Mode] ASL detection stopped — no hand detected.")
            asl_active = False

        if gesture != "Unknown" and gesture != last_printed:
            print(f"Gesture: {gesture}")
            last_printed = gesture

        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
