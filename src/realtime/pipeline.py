import json
import time
import warnings
from pathlib import Path

# scikit-learn warns when the pickle's version differs from the installed
# version. The spot-check confirmed the model loads and predicts correctly,
# so silence the noise — but only this specific warning class.
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import cv2
import joblib
import numpy as np
import tensorflow as tf
import mediapipe as mp

# New mediapipe Tasks API. The legacy `mediapipe.solutions.face_mesh`
# was removed in mediapipe >= 0.10.21, but the underlying model is
# the same canonical 468-point face mesh, so landmark indices match
# the ones the SVM was trained on.
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


# =========================================================
# Paths — resolved relative to project root for portability
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # .../Drowsiness-Project

FEATURES_DIR    = BASE_DIR / "feature_extraction" / "features_output"
SCALER_PATH     = FEATURES_DIR / "scaler.pkl"
LABEL_MAP_PATH  = FEATURES_DIR / "label_map.json"
SVM_MODEL_PATH  = BASE_DIR / "model" / "svm_model.pkl"
LANDMARKER_PATH = BASE_DIR / "model" / "face_landmarker.task"


# =========================================================
# Feature extraction config — MUST match feature_extraction.ipynb
# =========================================================
IMG_SIZE         = 256
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
EYE_CROP_PADDING = 20
BRIGHTNESS_RANGE = (20, 250)

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


# =========================================================
# Load scaler, SVM, and label map produced by the notebook
# =========================================================
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
        raw_label_map = json.load(f)  # {"drowsy": 0, "non_drowsy": 1}

    id_to_label = {int(v): k for k, v in raw_label_map.items()}

    if "drowsy" not in raw_label_map:
        raise ValueError(
            f"label_map.json is missing the 'drowsy' key: {raw_label_map}"
        )
    drowsy_id = int(raw_label_map["drowsy"])

    return scaler_obj, svm_obj, id_to_label, drowsy_id


scaler, svm_model, LABEL_MAP, DROWSY_CLASS_ID = _load_artifacts()


# =========================================================
# MobileNetV2 feature extractor (frozen ImageNet weights)
# Wrapped in a tf.function so per-frame calls don't pay
# the .predict() Python overhead.
# =========================================================
_base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
_base_model.trainable = False

feature_extractor = Model(
    inputs=_base_model.input,
    outputs=GlobalAveragePooling2D()(_base_model.output),
)


@tf.function(reduce_retracing=True)
def _extract_deep(crop_tensor):
    return feature_extractor(crop_tensor, training=False)


# =========================================================
# Preprocessing — mirrors the notebook exactly
# =========================================================
def preprocess_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if not (BRIGHTNESS_RANGE[0] < gray.mean() < BRIGHTNESS_RANGE[1]):
        return None

    denoised = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=35, sigmaSpace=35)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.cvtColor(
        cv2.merge([clahe.apply(l), a, b]),
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
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C + 1e-6)


def _detect_landmarks(landmarker, image_bgr, timestamp_ms):
    """Run the FaceLandmarker on a BGR image. Returns a list of NormalizedLandmark
    or None if no face was detected."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def extract_eye_features(image_bgr, landmarker, timestamp_ms):
    h, w = image_bgr.shape[:2]
    lm = _detect_landmarks(landmarker, image_bgr, timestamp_ms)
    if lm is None:
        return None

    def get_pts(indices):
        return np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices])

    left_crop_pts  = get_pts(LEFT_EYE_CROP)
    right_crop_pts = get_pts(RIGHT_EYE_CROP)

    left_ear  = compute_ear(get_pts(LEFT_EYE_EAR))
    right_ear = compute_ear(get_pts(RIGHT_EYE_EAR))
    ear       = (left_ear + right_ear) / 2.0
    ear_diff  = abs(left_ear - right_ear)

    if not (0.1 < ear < 0.5):
        return None

    all_pts = np.concatenate([left_crop_pts, right_crop_pts])
    x1 = max(0, np.min(all_pts[:, 0]) - EYE_CROP_PADDING)
    y1 = max(0, np.min(all_pts[:, 1]) - EYE_CROP_PADDING)
    x2 = min(w, np.max(all_pts[:, 0]) + EYE_CROP_PADDING)
    y2 = min(h, np.max(all_pts[:, 1]) + EYE_CROP_PADDING)

    if (x2 - x1) < 60:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return crop, float(ear), float(ear_diff)


def extract_features(frame_bgr, landmarker, timestamp_ms):
    processed = preprocess_image(frame_bgr)
    if processed is None:
        return None

    result = extract_eye_features(processed, landmarker, timestamp_ms)
    if result is None:
        return None

    crop, ear, ear_diff = result

    crop_batch = np.expand_dims(crop.astype(np.float32), axis=0)
    crop_batch = preprocess_input(crop_batch)

    # Direct __call__ via tf.function — much faster than .predict() per frame.
    deep_features = _extract_deep(tf.convert_to_tensor(crop_batch)).numpy()[0]

    feature_vector = np.concatenate([
        deep_features.astype(np.float32),
        np.array([ear, ear_diff], dtype=np.float32),
    ])
    return feature_vector, ear, ear_diff


def _build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


class DrowsinessPipeline:
    """
    Wraps preprocessing + landmark + MobileNet + scaler + SVM.

    .process(frame_bgr) returns a dict:
        {
            "prediction": int,    # raw class id from the SVM
            "label":      str,    # canonical label, e.g. "drowsy"
            "is_drowsy":  bool,   # True iff prediction == DROWSY_CLASS_ID
            "ear":        float,
            "ear_diff":   float,
        }
    or None if no valid face / frame skipped.
    """

    def __init__(self):
        self.landmarker = _build_landmarker()
        # Monotonically increasing timestamp required by VIDEO running mode.
        self._t0_ns = time.monotonic_ns()

    def _next_timestamp_ms(self):
        return (time.monotonic_ns() - self._t0_ns) // 1_000_000

    def process(self, frame_bgr):
        feats = extract_features(frame_bgr, self.landmarker,
                                 self._next_timestamp_ms())
        if feats is None:
            return None

        feature_vector, ear, ear_diff = feats

        X = scaler.transform(feature_vector.reshape(1, -1))
        prediction = int(svm_model.predict(X)[0])
        label = LABEL_MAP.get(prediction, str(prediction))

        return {
            "prediction": prediction,
            "label":      label,
            "is_drowsy":  prediction == DROWSY_CLASS_ID,
            "ear":        ear,
            "ear_diff":   ear_diff,
        }

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
