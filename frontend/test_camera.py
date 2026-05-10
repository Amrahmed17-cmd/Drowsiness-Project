import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # open default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
