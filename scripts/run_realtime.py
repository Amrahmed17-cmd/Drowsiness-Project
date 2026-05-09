import cv2
from collections import deque

from src.realtime.camera import read_frame, close_camera
from src.realtime.pipeline import DrowsinessPipeline


pipeline = DrowsinessPipeline()
pred_buffer = deque(maxlen=10)

while True:

    frame = read_frame()

    if frame is None:
        break

    result = pipeline.process(frame)

    if result is None:
        continue

    prediction, label = result

    pred_buffer.append(prediction)

    smoothed_prediction = 1 if sum(pred_buffer) / len(pred_buffer) > 0.5 else 0
    label = "drowsy" if smoothed_prediction == 1 else "non_drowsy"
    prediction = smoothed_prediction

    print(f"Prediction: {prediction}")
    print(f"State: {label}")

    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)

    cv2.putText(
        frame,
        label,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        "Press Q or ESC to quit",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break


close_camera()
cv2.destroyAllWindows()