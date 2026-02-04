import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# 2. File Setup
DATA_FILE = 'hand_signs_data.csv'

# Check if file exists; if not, create it with headers
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create headers: label, x0, y0, z0, x1, y1, z1 ... x20, y20, z20
        headers = ['label']
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(headers)

# 3. Input: Ask the user what sign they are recording
sign_name = "Hello"  # Hardcoded for testing purposes (Change as needed for different signs)
print(f"Press 's' to save a frame. Press 'q' to quit.")

cap = cv2.VideoCapture(1)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # PREPARE DATA TO SAVE
            # We flatten the landmarks into a single list: [x0, y0, z0, x1, y1, z1...]
            row = [sign_name]
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            # ON KEY PRESS 's', SAVE THE DATA
            key = cv2.waitKey(1)
            if key == ord('s'):
                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                counter += 1
                print(f"Captured sample {counter} for '{sign_name}'")

    # Display text on screen
    cv2.putText(frame, f"Sign: {sign_name} | Samples: {counter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Data Collection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()