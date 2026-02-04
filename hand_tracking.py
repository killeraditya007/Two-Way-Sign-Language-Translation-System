import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands
# This helps us draw the connections (lines) between the joints
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 2. Open the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

# 3. Setup the Hands Model
# min_detection_confidence: How sure the AI must be to say "That's a hand"
# min_tracking_confidence: How sure it must be to keep tracking the same hand
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    print("Webcam opened. Press 'q' to exit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 4. Pre-process the image
        # MediaPipe needs RGB images, but OpenCV gives us BGR. We must convert it.
        # We also mark it as not writeable to improve performance.
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 5. The Magic Happens Here (Detection)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for display

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # This draws the red dots and green lines
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # OPTIONAL: Print the coordinates of the Index Finger Tip (Landmark 8)
                # This is just to show you that the data is being read.
                # index_finger_tip = hand_landmarks.landmark[8]
                # print(f"Index Tip: {index_finger_tip.x}, {index_finger_tip.y}")

        # 6. Show the result in a window
        cv2.imshow('ASL Project - Hand Tracking', image)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()