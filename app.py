import cv2
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    # Your image processing code here
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()