import cv2

camera = cv2.VideoCapture(0)


def read_frame():
    success, frame = camera.read()
    if success:
        return frame
    return None


def close_camera():
    camera.release()
    cv2.destroyAllWindows()