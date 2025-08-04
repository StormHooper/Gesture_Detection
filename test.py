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

# For webcam input (no CAP_DSHOW on macOS)
cap = cv2.VideoCapture(0)  # <-- This is the only change

frame_count = 0  # Frame counter for yaw reporting

def calculate_yaw_angle(lm):
    # Use wrist (0), index_mcp (5), pinky_mcp (17)
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    # Calculate vector between index and pinky MCP joints
    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y

    # Angle in radians then convert to degrees
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    print("Press 'q' to quit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert the image to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(image_rgb)

        gesture = "Unknown"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                lm = hand_landmarks.landmark

                # Yaw angle reporting every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    yaw_angle = calculate_yaw_angle(lm)
                    print(f"Hand Z Angle: {yaw_angle:.2f}")

                # Finger tip landmarks (thumb is more complex)
                tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
                fingers = []

                for tip in tips:
                    fingers.append(lm[tip].y < lm[tip - 2].y)

                # Thumb (check x instead of y because it's sideways)
                thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
                fingers.insert(0, thumb_up)

                # Interpret fingers up and number of them
                gesture = ""
                number_of_fingers = 0
                if fingers[0] == True:
                    gesture = "Thumb"
                    number_of_fingers += 1
                if fingers[1] == True:
                    if gesture == "":
                        gesture = "Pointer"
                    else:
                        gesture += ", Pointer"
                    number_of_fingers += 1
                if fingers[2] == True:
                    if gesture == "":
                        gesture = "Middle"
                    else:
                        gesture += ", Middle"
                    number_of_fingers += 1
                if fingers[3] == True:
                    if gesture == "":
                        gesture == "Ring"
                    else:
                        gesture += ", Ring"
                    number_of_fingers += 1
                if fingers[4] == True:
                    if gesture == "":
                        gesture == "Pinky"
                    else:
                        gesture += ", Pinky"
                    number_of_fingers += 1
                if gesture == "":
                    number_of_fingers = 0
                
                if gesture != previous_gesture:
                    print(f"Gesture: {gesture} Number of Fingers: {str(number_of_fingers)}")
                    previous_gesture = gesture

        # Show image
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
