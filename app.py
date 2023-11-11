import cv2
import numpy as np
import mediapipe as mp 
import serial
import time
cap = cv2.VideoCapture(0) # dont change this 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

ser = serial.Serial('COM3', 9600)
time_threshold = 0.2
closed_fist_start_time = None
closed_fist_threshold = 0.81  # Adjust as needed
open_hand_threshold = 0.81
while True:
    ret, frame = cap.read()
    # Your image processing code here
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        middle_finger_y = (landmarks[12].y + landmarks[16].y) / 2
        ring_finger_y = (landmarks[16].y + landmarks[20].y) / 2
        thumb_tip_y = (landmarks[4].y + landmarks[8].y) / 2
        thumb_base_y = landmarks[2].y

        thumb_tip = (landmarks[4].x, landmarks[4].y)
        pinky_base = (landmarks[17].x, landmarks[17].y)
        distance_thumb_pinky = ((thumb_tip[0] - pinky_base[0])**2 + (thumb_tip[1] - pinky_base[1])**2)**0.5
        if(
            middle_finger_y < closed_fist_threshold and
            ring_finger_y < closed_fist_threshold and
            thumb_tip_y > closed_fist_threshold and
            thumb_base_y > closed_fist_threshold
        ):
            ser.write(b'1')
            cv2.putText(frame, 'Closed Fist Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If the hand is not in a closed fist position, reset the start time
            closed_fist_start_time = None
        # Check if the hand is open based on distance between thumb tip and pinky base
        if distance_thumb_pinky > open_hand_threshold:
            # Display text for open hand
            cv2.putText(frame, 'Open Hand Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ser.close()
cap.release()
cv2.destroyAllWindows()