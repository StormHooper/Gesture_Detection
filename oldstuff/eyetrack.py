import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    while True:
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Iris landmarks: 468-473 (left eye), 474-479 (right eye)
                iris_left = face_landmarks.landmark[468]
                iris_right = face_landmarks.landmark[473]
                # Estimate gaze direction by relative position of iris

        cv2.imshow("Eye Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
