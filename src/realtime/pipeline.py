import json
import math
import time
import warnings
from pathlib import Path

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # silence pickle version warning

import cv2
import joblib
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


BASE_DIR = Path(__file__).resolve().parents[2]   # project root

FEATURES_DIR    = BASE_DIR / "feature_extraction" / "features_output"
SCALER_PATH     = FEATURES_DIR / "scaler.pkl"
LABEL_MAP_PATH  = FEATURES_DIR / "label_map.json"
SVM_MODEL_PATH  = BASE_DIR / "model" / "svm_model.pkl"
LANDMARKER_PATH = BASE_DIR / "model" / "face_landmarker.task"


IMG_SIZE         = 256          # MobileNetV2 input size
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
EYE_CROP_PADDING = 20           # px of margin around eye landmarks
BRIGHTNESS_RANGE = (20, 250)    # reject too dark/bright frames

LEFT_EYE_CROP = [
    362, 382, 381, 380, 374, 373, 390,
    249, 263, 466, 388, 387, 386, 385, 384, 398,
]
RIGHT_EYE_CROP = [
    33, 7, 163, 144, 145, 153, 154,
    155, 133, 173, 157, 158, 159, 160, 161, 246,
]

LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]


CALIBRATION_FRAMES        = 30      # ~2s of frames to learn neutral head pose

HEAD_OFF_AXIS_DELTA_DEG   = 15.0    # threshold for "head off-axis" per frame

HEAD_OFF_AXIS_HOLD_FRAMES = 22      # consecutive off-axis frames to vote drowsy


def _load_artifacts():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"scaler.pkl not found at {SCALER_PATH}. "
            "Run feature_extraction.ipynb first."
        )
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"label_map.json not found at {LABEL_MAP_PATH}. "
            "Run feature_extraction.ipynb first."
        )
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"svm_model.pkl not found at {SVM_MODEL_PATH}. "
            "Train the SVM before running real-time inference."
        )
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            f"face_landmarker.task not found at {LANDMARKER_PATH}. "
            "Download it with:\n"
            "  curl -L -o model/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )

    scaler_obj = joblib.load(SCALER_PATH)
    svm_obj    = joblib.load(SVM_MODEL_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw_label_map = json.load(f)

    id_to_label = {int(v): k for k, v in raw_label_map.items()}  # invert label map

    if "drowsy" not in raw_label_map:
        raise ValueError(
            f"label_map.json is missing the 'drowsy' key: {raw_label_map}"
        )
    drowsy_id = int(raw_label_map["drowsy"])

    return scaler_obj, svm_obj, id_to_label, drowsy_id


scaler, svm_model, LABEL_MAP, DROWSY_CLASS_ID = _load_artifacts()  # load once at import


_base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
_base_model.trainable = False  # frozen

feature_extractor = Model(
    inputs=_base_model.input,
    outputs=GlobalAveragePooling2D()(_base_model.output),  # 1280-D embedding
)


@tf.function(reduce_retracing=True)  # avoid per-frame Python overhead
def _extract_deep(crop_tensor):
    return feature_extractor(crop_tensor, training=False)


def preprocess_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if not (BRIGHTNESS_RANGE[0] < gray.mean() < BRIGHTNESS_RANGE[1]):
        return None  # skip too dark/bright frames

    denoised = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=35, sigmaSpace=35)  # edge-preserving smooth

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.cvtColor(
        cv2.merge([clahe.apply(l), a, b]),  # CLAHE on L only
        cv2.COLOR_LAB2BGR,
    )

    enhanced = cv2.normalize(
        enhanced,
        np.zeros_like(enhanced),
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    return enhanced


def compute_ear(points):
    A = np.linalg.norm(points[1] - points[5])    # vertical 1
    B = np.linalg.norm(points[2] - points[4])    # vertical 2
    C = np.linalg.norm(points[0] - points[3])    # horizontal
    return (A + B) / (2.0 * C + 1e-6)            # eye aspect ratio


def _detect_landmarks(landmarker, image_bgr, timestamp_ms):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    if not result.face_landmarks:
        return None, None  # no face detected
    landmarks = result.face_landmarks[0]
    matrix = (
        np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)  # 4x4 face->camera
        if getattr(result, "facial_transformation_matrixes", None)
        else None
    )
    return landmarks, matrix


def _matrix_to_euler_deg(matrix):
    if matrix is None:
        return None
    R = matrix[:3, :3]                                        # rotation block
    sy = math.sqrt(float(R[0, 0]) ** 2 + float(R[1, 0]) ** 2)
    if sy > 1e-6:
        pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw   = math.atan2(-float(R[2, 0]), sy)
        roll  = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:                                                     # gimbal-lock fallback
        pitch = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        yaw   = math.atan2(-float(R[2, 0]), sy)
        roll  = 0.0
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def extract_eye_features(image_bgr, landmarker, timestamp_ms):
    h, w = image_bgr.shape[:2]
    lm, face_matrix = _detect_landmarks(landmarker, image_bgr, timestamp_ms)
    if lm is None:
        return None

    def get_pts(indices):
        return np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices])  # normalize -> px

    left_crop_pts  = get_pts(LEFT_EYE_CROP)
    right_crop_pts = get_pts(RIGHT_EYE_CROP)

    left_ear  = compute_ear(get_pts(LEFT_EYE_EAR))
    right_ear = compute_ear(get_pts(RIGHT_EYE_EAR))
    ear       = (left_ear + right_ear) / 2.0    # mean EAR
    ear_diff  = abs(left_ear - right_ear)       # asymmetry between eyes

    if not (0.1 < ear < 0.5):
        return None    # implausible EAR -> bad detection

    all_pts = np.concatenate([left_crop_pts, right_crop_pts])
    x1 = max(0, np.min(all_pts[:, 0]) - EYE_CROP_PADDING)
    y1 = max(0, np.min(all_pts[:, 1]) - EYE_CROP_PADDING)
    x2 = min(w, np.max(all_pts[:, 0]) + EYE_CROP_PADDING)
    y2 = min(h, np.max(all_pts[:, 1]) + EYE_CROP_PADDING)

    if (x2 - x1) < 60:
        return None    # crop too small

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    head_pose = _matrix_to_euler_deg(face_matrix)  # (pitch, yaw, roll) deg
    return crop, float(ear), float(ear_diff), head_pose


def extract_features(frame_bgr, landmarker, timestamp_ms):
    processed = preprocess_image(frame_bgr)
    if processed is None:
        return None

    result = extract_eye_features(processed, landmarker, timestamp_ms)
    if result is None:
        return None

    crop, ear, ear_diff, head_pose = result

    crop_batch = np.expand_dims(crop.astype(np.float32), axis=0)
    crop_batch = preprocess_input(crop_batch)               # MobileNetV2 normalization

    deep_features = _extract_deep(tf.convert_to_tensor(crop_batch)).numpy()[0]  # 1280-D

    feature_vector = np.concatenate([
        deep_features.astype(np.float32),
        np.array([ear, ear_diff], dtype=np.float32),         # append EAR features
    ])
    return feature_vector, ear, ear_diff, head_pose


def _build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.3,                       # low to keep tracking through small turns
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,        # needed for head pose
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


class DrowsinessPipeline:

    def __init__(self, calibration_frames: int = CALIBRATION_FRAMES):
        self.landmarker = _build_landmarker()
        self._t0_ns = time.monotonic_ns()                  # monotonic clock origin

        self._calibration_target  = max(1, int(calibration_frames))
        self._calibration_samples = []
        self._head_baseline       = None                   # set after calibration
        self._head_off_streak     = 0

    def _next_timestamp_ms(self):
        return (time.monotonic_ns() - self._t0_ns) // 1_000_000  # MediaPipe needs monotonic ms

    def reset_calibration(self):
        self._calibration_samples = []
        self._head_baseline       = None
        self._head_off_streak     = 0

    def _update_head_channel(self, head_pose):
        if head_pose is None:
            calibrating = self._head_baseline is None
            progress    = (
                len(self._calibration_samples) / self._calibration_target
                if calibrating else 1.0
            )
            return None, False, False, calibrating, progress

        if self._head_baseline is None:                    # phase 1: collect baseline
            self._calibration_samples.append(head_pose)
            progress = len(self._calibration_samples) / self._calibration_target
            if len(self._calibration_samples) >= self._calibration_target:
                arr = np.asarray(self._calibration_samples, dtype=np.float32)
                self._head_baseline = tuple(float(x) for x in np.median(arr, axis=0))  # robust to outliers
                progress = 1.0
            return None, False, False, self._head_baseline is None, min(1.0, progress)

        rel = tuple(float(c - b) for c, b in zip(head_pose, self._head_baseline))  # deviation
        max_dev = max(abs(rel[0]), abs(rel[1]), abs(rel[2]))         # any-axis magnitude
        is_off  = max_dev >= HEAD_OFF_AXIS_DELTA_DEG

        if is_off:
            self._head_off_streak += 1
        else:
            self._head_off_streak = 0                                # reset on aligned frame

        is_head_drowsy = self._head_off_streak >= HEAD_OFF_AXIS_HOLD_FRAMES
        return rel, is_off, is_head_drowsy, False, 1.0

    def process(self, frame_bgr):
        feats = extract_features(frame_bgr, self.landmarker,
                                 self._next_timestamp_ms())
        if feats is None:
            return None

        feature_vector, ear, ear_diff, head_pose = feats

        X = scaler.transform(feature_vector.reshape(1, -1))   # match training scaling
        prediction = int(svm_model.predict(X)[0])             # SVM eye-channel decision
        label = LABEL_MAP.get(prediction, str(prediction))

        (rel, is_head_off, is_head_drowsy,
         calibrating, calib_progress) = self._update_head_channel(head_pose)

        return {
            "prediction":            prediction,
            "label":                 label,
            "is_drowsy":             prediction == DROWSY_CLASS_ID,
            "ear":                   ear,
            "ear_diff":              ear_diff,
            "head_pose":             head_pose,
            "head_pose_relative":    rel,
            "is_head_off_axis":      is_head_off,
            "head_off_streak":       self._head_off_streak,
            "is_head_drowsy":        is_head_drowsy,
            "calibrating":           calibrating,
            "calibration_progress":  calib_progress,
        }

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
