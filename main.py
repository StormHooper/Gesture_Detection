import cv2
import mediapipe as mp
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

                # Finger tip landmarks (thumb is more complex)
                tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
                fingers = []

                for tip in tips:
                    # Check if the tip is above the lower joint
                    fingers.append(lm[tip].y < lm[tip - 2].y)

                # Thumb (check x instead of y because it's sideways)
                thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
                fingers.insert(0, thumb_up)

                # Interpret gesture
                if fingers == [False, False, False, False, False]:
                    gesture = "Fist"
                elif fingers == [True, True, True, True, True]:
                    gesture = "Open Palm"
                elif fingers == [False, True, False, False, False]:
                    gesture = "Pointing"
                elif fingers == [False, True, True, False, False]:
                    gesture = "Peace"
                elif fingers == [False, False, True, False, False]:
                    gesture = "Bird"
                elif fingers == [True, False, False, False, True]:
                    gesture = "Phone"
                elif fingers == [True, False, False, False, False]:
                    gesture = "Thumbs Up"
                elif fingers == [False, False, False, False, True]:
                    gesture = "Other Bird"

                print(f"Gesture: {gesture}")

        # Show image
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()