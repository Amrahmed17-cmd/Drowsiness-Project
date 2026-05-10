"""
Real-time drowsiness detection.

Run from the project root:
    python -m scripts.run_realtime

Label convention (must match feature_extraction.ipynb):
    drowsy     -> class 0
    non_drowsy -> class 1
"""

import time
from collections import deque

import cv2

from src.realtime.camera   import read_frame, close_camera
from src.realtime.pipeline import DrowsinessPipeline


# How many recent frames to smooth over.
SMOOTH_WINDOW = 10
# Fraction of frames in the window that must be "drowsy" to flag drowsy.
DROWSY_THRESHOLD = 0.5


def draw_hud(frame, *, smoothed_label, raw_label, drowsy_score, fps,
             ear=None, ear_diff=None):
    """Overlay the current state on the frame."""
    is_drowsy = smoothed_label == "drowsy"
    color     = (0, 0, 255) if is_drowsy else (0, 255, 0)
    state     = "DROWSY" if is_drowsy else "AWAKE"

    # status banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, thickness=-1)
    cv2.putText(
        frame, f"State: {state}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )

    info_lines = [
        f"Raw: {raw_label}   Smoothed: {smoothed_label}",
        f"Drowsy score: {drowsy_score:.2f}   FPS: {fps:.1f}",
    ]
    if ear is not None and ear_diff is not None:
        info_lines.append(f"EAR: {ear:.3f}   EAR diff: {ear_diff:.3f}")
    info_lines.append("Press Q or ESC to quit")

    y = 70
    for line in info_lines:
        cv2.putText(
            frame, line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        y += 28


def main():
    pipeline = DrowsinessPipeline()
    drowsy_buffer = deque(maxlen=SMOOTH_WINDOW)  # 1 = drowsy frame, 0 = non_drowsy

    last_t = time.time()
    fps    = 0.0

    try:
        while True:
            frame = read_frame()
            if frame is None:
                break

            now    = time.time()
            dt     = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # exp. moving avg.

            result = pipeline.process(frame)

            if result is None:
                cv2.putText(
                    frame, "No valid face / frame skipped",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            drowsy_buffer.append(1 if result["is_drowsy"] else 0)
            drowsy_score   = sum(drowsy_buffer) / len(drowsy_buffer)
            smoothed_label = "drowsy" if drowsy_score > DROWSY_THRESHOLD else "non_drowsy"

            draw_hud(
                frame,
                smoothed_label=smoothed_label,
                raw_label=result["label"],
                drowsy_score=drowsy_score,
                fps=fps,
                ear=result["ear"],
                ear_diff=result["ear_diff"],
            )

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        pipeline.close()
        close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
