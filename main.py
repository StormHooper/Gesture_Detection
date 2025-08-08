import cv2, mediapipe as mp, os, time as t
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import platform

# Check OS
IS_WINDOWS = platform.system() == "Windows"

# Windows-only imports
if IS_WINDOWS:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    import screen_brightness_control as sbc
else:
    print("[Info] Non-Windows OS detected — audio/brightness control disabled.")

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if IS_WINDOWS else 0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load ASL model if available
model_path = Path("model/asl_knn_model.joblib")
encoder_path = Path("model/label_encoder.joblib")
if model_path.exists() and encoder_path.exists():
    asl_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("[Model] ASL recognition model loaded.")
else:
    asl_model = None
    label_encoder = None
    print("[Warning] No ASL model found. ASL recognition will be disabled.")

# State
active_read = False
cooldown_secs = 3.0
last_event_time = 0
last_printed = None
command_cooldown_secs = 1.5
last_command_time = 0
asl_active = False
last_asl_letter = None
asl_start_time = 0

# --- Actions ---
def brightness_up(step=10):
    if IS_WINDOWS:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(min(100, current + step), display=0)
            print(f"[Action] Brightness increased to {min(100, current + step)}%")
        except Exception as e:
            print(f"[Error] Brightness control failed: {e}")
    else:
        print("[Info] Brightness up not supported on macOS.")

def brightness_down(step=10):
    if IS_WINDOWS:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(max(0, current - step), display=0)
            print(f"[Action] Brightness decreased to {max(0, current - step)}%")
        except Exception as e:
            print(f"[Error] Brightness control failed: {e}")
    else:
        print("[Info] Brightness down not supported on macOS.")

def get_volume_interface():
    if IS_WINDOWS:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    return None

def toggle_mute():
    if IS_WINDOWS:
        volume = get_volume_interface()
        volume.SetMute(0 if volume.GetMute() else 1, None)
    else:
        print("[Info] Mute toggle not supported on macOS.")

def volume_up(step=0.1):
    if IS_WINDOWS:
        volume = get_volume_interface()
        current = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(min(1.0, current + step), None)
    else:
        print("[Info] Volume up not supported on macOS.")

def volume_down(step=0.1):
    if IS_WINDOWS:
        volume = get_volume_interface()
        current = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(max(0.0, current - step), None)
    else:
        print("[Info] Volume down not supported on macOS.")

# Gesture Actions
def do_open_palm_action():
    print("[Action] Turning lights ON (simulated).")

def do_pointing_action():
    print("[Action] You pointed — increasing volume by 10.")
    volume_up()

def do_peace_action():
    print("[Action] Peace gesture — muting audio...")
    toggle_mute()

def do_bird_action():
    print("[Action] Middle finger — rude! Logging it as a joke.")

def do_phone_action():
    print("[Action] Phone - lowering the brightness by 10.")
    brightness_down()

def do_thumbs_up_action():
    print("[Action] Thumbs up — upping the brightness by 10.")
    brightness_up()

def do_other_bird_action():
    print("[Action] Pinky - Lowering volume by 10.")
    volume_down()

# Main loop
with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

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
                    active_read = True
                    last_event_time = now
                    print("Fist detected – command mode will start in "
                          f"{cooldown_secs:.0f} s…")
                    break

                if active_read and now - last_event_time < cooldown_secs:
                    break

                if active_read and not asl_active:
                    if fingers == [True]*5:
                        gesture, active_read = "Open Palm", False
                        do_open_palm_action()
                        last_command_time = now
                    elif fingers == [False, True, False, False, False]:
                        gesture, active_read = "Pointing", False
                        do_pointing_action()
                        last_command_time = now
                    elif fingers == [False, True, True, False, False]:
                        gesture, active_read = "Peace", False
                        do_peace_action()
                        last_command_time = now
                    elif fingers == [False, False, True, False, False]:
                        gesture, active_read = "Bird", False
                        do_bird_action()
                        last_command_time = now
                    elif fingers == [True, False, False, False, True]:
                        gesture, active_read = "Phone", False
                        do_phone_action()
                        last_command_time = now
                    elif fingers == [True, False, False, False, False]:
                        gesture, active_read = "Thumbs Up", False
                        do_thumbs_up_action()
                        last_command_time = now
                    elif fingers == [False, False, False, False, True]:
                        gesture, active_read = "Other Bird", False
                        do_other_bird_action()
                        last_command_time = now
                    elif fingers == [True, True, True, False, False]:
                        gesture, active_read = "OK", False
                        asl_active = True
                        asl_start_time = now
                        last_command_time = now
                        print("[Mode] ASL detection activated.")
        else:
            if asl_active:
                print("[Mode] ASL detection stopped — no hand detected.")
            asl_active = False

        if gesture != "Unknown" and gesture != last_printed:
            print(f"Gesture: {gesture}")
            last_printed = gesture

        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()