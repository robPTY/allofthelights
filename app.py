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

        # Extract landmarks for the thumb, index finger, and middle finger
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        pinky_tip = landmarks[20]

        # Calculate distances between thumb tip and index/middle fingertips
        distance_thumb_index = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        distance_thumb_middle = ((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)**0.5
        distance_thumb_pinky = ((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)**0.5
         # Calculate the x-coordinate difference between thumb tip and index fingertip
        x_diff_thumb_index = thumb_tip.x - index_tip.x

        # Define a threshold for the circle gesture
        circle_threshold = 0.05

        # Define a threshold for the 'ok' sign
        ok_sign_threshold = 0.05

        # Define a threshold for the thumbs-up gesture
        thumbs_up_threshold = 0.1

        # Define a threshold for the thumbs-down gesture
        thumbs_down_threshold = 0.05

        # Define a threshold for the left arrow gesture
        left_arrow_threshold = -0.1

        # Define a threshold for the right arrow gesture
        right_arrow_threshold = 0.1

        if (abs(distance_thumb_index - distance_thumb_pinky) < circle_threshold):
            cv2.putText(frame, 'Turn LIGHT off', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ser.write(b'2')
            print("turning off")
        elif(
            distance_thumb_index < ok_sign_threshold and 
            distance_thumb_middle < ok_sign_threshold
        ):
            cv2.putText(frame,'Turn light on', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif(distance_thumb_index < thumbs_up_threshold):
            cv2.putText(frame,'Turn LIGHT on', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ser.write(b'1')
        elif(distance_thumb_index > thumbs_down_threshold):
            cv2.putText(frame,'Turn BRIGHTNESS down', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ser.write(b'3')
        elif(x_diff_thumb_index < left_arrow_threshold):
            cv2.putText(frame,'Next color', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ser.write(b'3')
        elif(x_diff_thumb_index > right_arrow_threshold):
            cv2.putText(frame,'Previous color', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If the hand is not any defined positions, reset the start time
            closed_fist_start_time = None
        # Check if the hand is open based on distance between thumb tip and pinky base
        if distance_thumb_pinky > open_hand_threshold:
            # Display text for open hand
            cv2.putText(frame, 'Open Hand Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ser.close()
cap.release()
cv2.destroyAllWindows()
