import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load your trained model and label encoder
model = joblib.load("model/asl_knn_model.joblib")
encoder = joblib.load("model/label_encoder.joblib")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("[Error] Cannot access webcam.")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    print("Press ';' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Empty frame, skipping.")
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        prediction = "No Hand Detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                X = np.array(landmark_list).reshape(1, -1)
                y_pred = model.predict(X)
                prediction = encoder.inverse_transform(y_pred)[0]

        cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Live Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord(';'):
            break

cap.release()
cv2.destroyAllWindows()
