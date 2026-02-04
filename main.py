import cv2
import mediapipe as mp
import joblib
import numpy as np

# 1. Load the Trained Model
print("Loading model...")
model = joblib.load('sign_language_model.p')
print("Model loaded successfully!")

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# 3. Open Webcam
cap = cv2.VideoCapture(1)

print("Starting Real-Time Translation... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mirror the frame so it looks natural
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # A. Draw the hand skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # B. Extract the Data (exactly like we did in collection)
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.extend([landmark.x, landmark.y, landmark.z])

            # C. Ask the Model for a Prediction
            # We must wrap the data in a list [] because the model expects a batch of inputs
            prediction = model.predict([data_aux])
            predicted_character = prediction[0]

            # D. Display the Result
            # Draw a box and the text
            cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1) # Black background box
            cv2.putText(frame, predicted_character, (20, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow('Sign Language Translator', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()