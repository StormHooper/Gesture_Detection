import cv2
import mediapipe as mp
import os
import math

# Prevent Spam
previous_gesture = ""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For webcam input (macOS compatible)
cap = cv2.VideoCapture(0)

frame_count = 0  # Frame counter for yaw reporting
angle_delay = 15  # Report yaw angle every 30 frames
angle_detection_active = False  # Toggle for 'a' key

def calculate_yaw_angle(lm):
    # Use wrist (0), index_mcp (5), pinky_mcp (17)
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y

    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    print("Press 'a' to toggle angle detection.")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark

                # --- Angle detection only if toggled ---
                if angle_detection_active:
                    frame_count += 1
                    if frame_count % angle_delay == 0:
                        yaw_angle = calculate_yaw_angle(lm)
                        print(f"Hand Z Angle: {yaw_angle:.2f}")

                # --- Gesture detection (always runs) ---
                tips = [8, 12, 16, 20]
                fingers = [lm[tip].y < lm[tip - 2].y for tip in tips]
                thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
                fingers.insert(0, thumb_up)

                gesture = ""
                number_of_fingers = 0
                if fingers[0]:
                    gesture = "Thumb"
                    number_of_fingers += 1
                if fingers[1]:
                    gesture = f"{gesture+', ' if gesture else ''}Pointer"
                    number_of_fingers += 1
                if fingers[2]:
                    gesture = f"{gesture+', ' if gesture else ''}Middle"
                    number_of_fingers += 1
                if fingers[3]:
                    gesture = f"{gesture+', ' if gesture else ''}Ring"
                    number_of_fingers += 1
                if fingers[4]:
                    gesture = f"{gesture+', ' if gesture else ''}Pinky"
                    number_of_fingers += 1
                if gesture == "":
                    number_of_fingers = 0

                if gesture != previous_gesture:
                    print(f"Gesture: {gesture} Number of Fingers: {number_of_fingers}")
                    previous_gesture = gesture

        cv2.imshow('Gesture Recognition', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            angle_detection_active = not angle_detection_active
            print(f"[Angle Detection] {'Activated' if angle_detection_active else 'Deactivated'}")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
