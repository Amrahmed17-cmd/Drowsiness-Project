
import time
from collections import deque

import cv2

from src.realtime.camera   import read_frame, close_camera
from src.realtime.pipeline import DrowsinessPipeline


SMOOTH_WINDOW     = 30      # frames to smooth over (~2s @ 15fps)
ENTER_DROWSY_FRAC = 0.55    # vote ratio to enter Drowsy
EXIT_DROWSY_FRAC  = 0.30    # vote ratio to leave Drowsy (hysteresis)


def draw_hud(
    frame, *,
    state_label,
    raw_label,
    drowsy_score,
    fps,
    ear=None,
    ear_diff=None,
    head_pose_rel=None,
    head_status=None,
    calib_progress=None,
):
    if state_label == "DROWSY":
        color = (0, 0, 255)         # red
    elif state_label == "AWAKE":
        color = (0, 255, 0)         # green
    else:
        color = (255, 165, 0)       # orange (calibrating)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, thickness=-1)
    banner = f"State: {state_label}"
    if state_label == "CALIBRATING" and calib_progress is not None:
        banner += f"   {int(round(calib_progress * 100))}%"
    cv2.putText(
        frame, banner,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )

    info_lines = [
        f"Raw: {raw_label}   Score: {drowsy_score:.2f}   FPS: {fps:.1f}",
    ]
    if ear is not None and ear_diff is not None:
        info_lines.append(f"EAR: {ear:.3f}   EAR diff: {ear_diff:.3f}")
    if head_pose_rel is not None:
        p, y, r = head_pose_rel
        info_lines.append(
            f"Head: pitch {p:+5.1f}  yaw {y:+5.1f}  roll {r:+5.1f}  [{head_status}]"
        )
    elif head_status is not None:
        info_lines.append(f"Head: {head_status}")
    info_lines.append("Press Q or ESC to quit")

    yy = 70
    for line in info_lines:
        cv2.putText(
            frame, line,
            (10, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        yy += 28


def _head_status(result):
    if result["calibrating"]:
        return "CALIBRATING"
    if result["is_head_drowsy"]:
        return "OFF-AXIS"
    if result["head_off_streak"] > 0:
        return "DRIFTING"
    return "ALIGNED"


def main():
    pipeline = DrowsinessPipeline()
    pipeline.reset_calibration()                 # fresh head-pose baseline

    drowsy_buffer = deque(maxlen=SMOOTH_WINDOW)  # rolling window of votes
    state_label   = "AWAKE"
    last_t        = time.time()
    fps           = 0.0
    calib_logged  = False

    try:
        while True:
            frame = read_frame()
            if frame is None:
                break

            now    = time.time()
            dt     = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)   # EMA-smoothed FPS

            result = pipeline.process(frame)         # eye + head-pose inference

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

            if result["calibrating"]:
                draw_hud(
                    frame,
                    state_label="CALIBRATING",
                    raw_label=result["label"],
                    drowsy_score=0.0,
                    fps=fps,
                    ear=result["ear"],
                    ear_diff=result["ear_diff"],
                    head_pose_rel=result["head_pose_relative"],
                    head_status=_head_status(result),
                    calib_progress=result["calibration_progress"],
                )
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            if not calib_logged:
                print("[run_realtime] Calibration complete — monitoring active.")
                calib_logged = True

            frame_drowsy = bool(result["is_drowsy"]) or bool(result["is_head_drowsy"])  # eye OR head vote
            drowsy_buffer.append(1 if frame_drowsy else 0)
            drowsy_score = sum(drowsy_buffer) / len(drowsy_buffer)  # smoothed fraction

            if state_label == "DROWSY":
                is_drowsy = drowsy_score >= EXIT_DROWSY_FRAC   # easier to stay
            else:
                is_drowsy = drowsy_score >= ENTER_DROWSY_FRAC  # harder to enter
            state_label = "DROWSY" if is_drowsy else "AWAKE"

            draw_hud(
                frame,
                state_label=state_label,
                raw_label=result["label"],
                drowsy_score=drowsy_score,
                fps=fps,
                ear=result["ear"],
                ear_diff=result["ear_diff"],
                head_pose_rel=result["head_pose_relative"],
                head_status=_head_status(result),
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
