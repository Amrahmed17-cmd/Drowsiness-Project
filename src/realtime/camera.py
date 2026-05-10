import sys
import cv2


CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30


_camera = None  # opened lazily on first read


def _open_camera():
    backends = []
    if sys.platform.startswith("win"):
        backends.append(cv2.CAP_DSHOW)  # DirectShow first on Windows
    backends.append(cv2.CAP_ANY)

    last_err = None
    for backend in backends:
        try:
            cam = cv2.VideoCapture(CAMERA_INDEX, backend)
        except Exception as exc:
            last_err = exc
            continue
        if cam is not None and cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cam.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
            cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # keep only the latest frame
            return cam
        if cam is not None:
            cam.release()

    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}. "
        f"Last error: {last_err!r}"
    )


def _get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = _open_camera()
    return _camera


def read_frame():
    cam = _get_camera()
    success, frame = cam.read()
    if not success or frame is None:
        return None
    return frame


def close_camera():
    global _camera
    if _camera is not None:
        if _camera.isOpened():
            _camera.release()
        _camera = None
