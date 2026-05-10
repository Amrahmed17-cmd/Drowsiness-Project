# Real-Time Drowsiness Detection System

<p align="center">
  <img alt="Python"     src="https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="MediaPipe"  src="https://img.shields.io/badge/MediaPipe-0.10.9-00897B?logo=google&logoColor=white">
  <img alt="OpenCV"     src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-SVC-F7931E?logo=scikit-learn&logoColor=white">
  <img alt="Streamlit"  src="https://img.shields.io/badge/Streamlit-GUI-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License"    src="https://img.shields.io/badge/license-MIT-green">
</p>

A two-channel driver-monitoring system that detects drowsiness in real time from a single webcam.
It combines a deep eye-region classifier (MobileNetV2 + EAR features → SVM) with a sustained head-pose deviation detector to deliver low-latency, low-flicker alerts with calibrated confidence.

---

## Table of Contents

- [Highlights](#highlights)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Configuration Reference](#configuration-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Highlights

- **Two-channel decision** — eye-region SVM **OR** sustained head-pose deviation
- **Per-session calibration** — learns the user's neutral head pose in ~2 s before arming
- **Hysteresis-stabilized output** — separate enter/exit thresholds kill flicker around the boundary
- **PERCLOS-style smoothing** — 30-frame rolling vote window (~2 s @ 15 fps)
- **Production-feel UI** — glassmorphism Streamlit HUD with live FPS, EAR, head pose, and event log
- **Audio alerts** — cooldown-gated WAV playback so the driver is alerted, not annoyed
- **CPU + GPU streaming feature extractor** — thread-pool preprocesses while the GPU runs MobileNetV2

---

## Demo

```
+-----------------------------+   +---------------------------+
|        LIVE FEED            |   |   DRIVER STATUS           |
|  +----+ REC                 |   |     [DROWSY]              |
|  |    |                     |   |                           |
|  | :) |  <--- webcam        |   |   FATIGUE SCORE  72/100   |
|  +----+                     |   |   [=========>      ]      |
|  15.3 FPS                   |   |                           |
+-----------------------------+   |   SESSION   00:42         |
                                  |   FRAMES    632           |
| EVENT LOG                       |   ALERTS    3             |
| 22:31:08  Drowsiness detected   |   HEAD      OFF-AXIS      |
| 22:30:55  Calibration complete  +---------------------------+
```

> Launch `python frontend/run_app.py` for the full Streamlit GUI, or `python -m scripts.run_realtime` for an OpenCV window.

---

## How It Works

### Architecture

```
Webcam frame (BGR)
        |
        v
+-----------------------+
|  Preprocessing         |  CLAHE on L-channel + bilateral denoise + brightness gate
+-----------------------+
        |
        v
+-----------------------+
|  MediaPipe FaceLandm.  |  468-pt mesh + 4x4 facial transform matrix
+-----------------------+
        |
        +-----------------------------+
        |                             |
        v                             v
+-----------------------+   +-----------------------+
|  Eye Channel           |   |  Head-Pose Channel    |
|  - Crop eye region     |   |  - Decompose to       |
|  - Compute EAR / diff  |   |    pitch / yaw / roll |
|  - MobileNetV2 -> 1280 |   |  - Deviation from     |
|  - Concat EAR features |   |    calibrated         |
|  - StandardScaler      |   |    baseline           |
|  - SVM (drowsy / not)  |   |  - Sustained off-axis |
+-----------------------+   +-----------------------+
        |                             |
        +--------------+--------------+
                       v
              Per-frame drowsy vote (eye OR head)
                       |
                       v
        Rolling window (30 frames) + Hysteresis
                       |
                       v
              State: AWAKE / DROWSY / CALIBRATING
                       |
                       v
            UI update + cooldown-gated audio alert
```

### Eye Channel (SVM)

1. **Face landmarks** are detected with MediaPipe FaceLandmarker (canonical 468-point mesh, video mode).
2. The **eye region** is cropped using 16 landmarks per eye plus a 20-px padding.
3. **EAR (Eye Aspect Ratio)** and **EAR difference** between eyes are computed from the canonical 6 EAR landmarks.
4. The eye crop is resized to 256×256, MobileNetV2-preprocessed, and pushed through a **frozen ImageNet MobileNetV2** to produce a 1280-D embedding.
5. The 1280-D embedding is concatenated with `[EAR, EAR_diff]` to form a **1282-D feature vector**.
6. A `StandardScaler` (fit on train) standardizes the features; an `SVC` (RBF kernel by default, `class_weight="balanced"`) outputs the per-frame `drowsy` / `non_drowsy` prediction.

### Head-Pose Channel

1. MediaPipe also returns the **face → camera 4×4 transformation matrix** per frame.
2. Its rotation block is decomposed into Euler angles `(pitch, yaw, roll)` in degrees.
3. For ~2 s after START the system collects the first 30 head-pose samples and stores their **median** as the per-session baseline (median is robust to single-frame jitter).
4. Each subsequent frame compares the current pose to the baseline. If the largest-magnitude axis deviates by `>= 15°` and stays off-axis for `>= 22 consecutive frames` (~1.5 s), the head channel votes **drowsy**.

### Smoothing & Hysteresis

A frame is considered drowsy if **either** channel votes drowsy. Each per-frame vote (`1` / `0`) is pushed into a 30-frame deque (`SMOOTH_WINDOW`), and the rolling **drowsy fraction** drives a Schmitt trigger:

| Transition          | Threshold                 | ~Time @ 15 fps |
|---------------------|---------------------------|----------------|
| Awake → Drowsy      | `score >= 0.55` (`ENTER`) | ~1.1 s         |
| Drowsy → Awake      | `score <= 0.30` (`EXIT`)  | ~0.6 s         |
| Dead-zone (no flip) | `0.30 < score < 0.55`     | —              |

Recovery is intentionally faster than commit, so the UI feels responsive when the driver wakes up while still being resistant to single-blink noise.

### Calibration

While the head-pose baseline is still being collected, the system reports `Calibrating`, **does not** push votes into the smoothing buffer, and **does not** raise alerts. A progress bar in the GUI shows calibration progress; on completion an "armed" event is emitted to the log.

---

## Project Structure

```
Drowsiness-Project/
├── datasets/
│   ├── clean_dataset/                   # final dataset, split into train/test
│   │   ├── train/{drowsy,non_drowsy}/
│   │   └── test/{drowsy,non_drowsy}/
│   ├── dataset_split.csv                # path,label,split mapping
│   └── data_acquisition.ipynb           # dataset assembly
│
├── feature_extraction/
│   ├── feature_extraction.ipynb         # MAIN: MobileNetV2 + EAR pipeline (CPU+GPU streaming)
│   ├── drowsiness_feature_extraction.ipynb  # ALT: pure MobileNetV2 (no EAR / no eye crop)
│   └── features_output/                 # X_*.npy, y_*.npy, scaler.pkl, label_map.json, ...
│
├── model/
│   ├── face_landmarker.task             # MediaPipe FaceLandmarker model (download once)
│   ├── svm_model.pkl                    # trained SVM (output of svm_training.ipynb)
│   ├── svm_training.ipynb               # grid-search SVM training
│   └── evaluation.ipynb                 # full metric suite (CM, ROC, AUC, etc.)
│
├── src/realtime/
│   ├── camera.py                        # cross-platform webcam helpers
│   └── pipeline.py                      # DrowsinessPipeline (eye + head channels)
│
├── scripts/
│   └── run_realtime.py                  # CLI runner with OpenCV HUD overlay
│
├── frontend/
│   ├── app.py                           # Streamlit HUD GUI (~950 lines)
│   ├── run_app.py                       # cross-cwd launcher with auto-config
│   ├── generate_alert.py                # synthesizes the alert WAV
│   └── test_camera.py                   # 10-line webcam smoke test
│
├── preprocessing.ipynb                  # initial dataset preprocessing
├── requirements.txt                     # pinned deps (TF 2.10 + MediaPipe 0.10.9)
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Installation

### Prerequisites

- **Windows 10/11** (the requirements pin is the last native-Windows GPU build of TensorFlow)
- **Python 3.10** (TF 2.10 does not support 3.11+)
- A working **webcam**
- *(Optional)* **NVIDIA GPU + CUDA 11.2 + cuDNN 8.1** for ~3-5× faster MobileNetV2 inference

### Setup

```bash
git clone https://github.com/<your-username>/Drowsiness-Project.git
cd Drowsiness-Project

python -m venv .venv
.venv\Scripts\activate                  # PowerShell / cmd
# source .venv/bin/activate              # bash / zsh

pip install -r requirements.txt
pip install streamlit pygame joblib scikit-learn pandas seaborn tqdm
```

### Download the MediaPipe FaceLandmarker model

The `face_landmarker.task` file is required by the realtime pipeline and must live at `model/face_landmarker.task` (~3.6 MB):

```bash
curl -L -o model/face_landmarker.task ^
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> If the file is missing the realtime pipeline will raise a `FileNotFoundError` with the same URL printed in the message.

### Quick environment check

```bash
python frontend/test_camera.py          # ESC to quit; verifies webcam works
```

---

## Quick Start

The repo already ships with a **trained SVM** (`model/svm_model.pkl`) and the matching **StandardScaler** + **label map** under `feature_extraction/features_output/`, so you can run inference immediately without retraining.

### Option A — Streamlit GUI (recommended)

```bash
python frontend/run_app.py              # default port 8501
python frontend/run_app.py --port 8888  # custom port
python frontend/run_app.py --no-browser # headless / server mode
```

Then click **START** to begin a session. The first ~2 s show `CALIBRATING` while the head-pose baseline is learned, after which the system is armed.

### Option B — OpenCV CLI runner

```bash
python -m scripts.run_realtime
```

Press `Q` or `ESC` to quit. The HUD overlays the current state, EAR, head pose, and FPS directly on the camera feed.

---

## Dataset

The project uses a curated drowsiness/non-drowsy face image dataset assembled in `datasets/data_acquisition.ipynb` (with deduplication report in `datasets/duplicates_report.csv`).

### Statistics

| Split | drowsy | non_drowsy | total      |
|-------|--------|------------|------------|
| Train | 11,458 | 10,962     | **22,420** |
| Test  | 2,865  | 2,740      | **5,605**  |

After feature extraction (with EAR / face-detection gating), the usable samples are:

| Split | samples | feature dim | skipped (no face / bad EAR) |
|-------|---------|-------------|------------------------------|
| Train | 22,205  | 1282        | 215                          |
| Test  | 5,568   | 1282        | 37                           |

See `feature_extraction/features_output/features_metadata.json` for the canonical extraction record.

### Folder layout expected by the notebooks

```
datasets/clean_dataset/
├── train/
│   ├── drowsy/         *.jpg
│   └── non_drowsy/     *.jpg
└── test/
    ├── drowsy/         *.jpg
    └── non_drowsy/     *.jpg
```

`LABEL_MAP = {"drowsy": 0, "non_drowsy": 1}` (the realtime pipeline reads this from `feature_extraction/features_output/label_map.json`).

---

## End-to-End Pipeline

### 1. Dataset preprocessing

Notebook: **`preprocessing.ipynb`**

- Reads `datasets/dataset_split.csv`
- Builds `tf.data` pipelines for train / test with on-the-fly resize, normalization, and (train-only) augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`)
- Sanity-checks one batch from each split

### 2. Feature extraction

Notebook: **`feature_extraction/feature_extraction.ipynb`** (the canonical pipeline used by the realtime system)

For each image:

1. Brightness gate (mean gray must be in `(20, 250)`) — rejects too-dark / too-bright frames
2. Bilateral denoise (`d=5, sigma=35`)
3. CLAHE on the L-channel of LAB (`clip=2.0`, `tile=8x8`)
4. MediaPipe FaceMesh landmarks
5. EAR + EAR-diff from the 6 canonical EAR landmarks per eye
6. Eye crop (16-landmark bounding box + 20-px padding) → resize to 256×256
7. Train-only random augmentation (flip / brightness / blur / rotation)
8. MobileNetV2 (frozen) → 1280-D embedding
9. Concatenate `[embedding, EAR, EAR_diff]` → **1282-D feature vector**
10. `StandardScaler.fit_transform` on train, `.transform` on test
11. Save `X_train.npy / X_test.npy / y_train.npy / y_test.npy / scaler.pkl / label_map.json / features_metadata.json` under `feature_extraction/features_output/`

The extraction uses a CPU `ThreadPoolExecutor` for OpenCV/MediaPipe work and streams chunks of size 1024 to MobileNetV2 on the GPU (`gpu_batch=128`).

> An alternative pipeline (`drowsiness_feature_extraction.ipynb`) extracts pure MobileNetV2 embeddings on full images — useful for ablation but **not** what the realtime system loads.

### 3. SVM training

Notebook: **`model/svm_training.ipynb`**

- Loads features + scaler from `feature_extraction/features_output/`
- 3-fold `StratifiedKFold` grid search over:
  ```python
  {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
  ```
- Picks the highest mean CV accuracy, refits on the full training set, and saves `model/svm_model.pkl` via `joblib.dump`

### 4. Evaluation

Notebook: **`model/evaluation.ipynb`**

Reports on the held-out test set:
- Accuracy, classification error, precision, recall, specificity
- Confusion matrix (with PNG export)
- ROC curve + AUC (with PNG export)
- TPR / FPR breakdown
- Train vs test accuracy bar chart (overfitting check)
- Null-accuracy baseline (majority-class)

### 5. Real-time inference

Driver class: **`src/realtime/pipeline.DrowsinessPipeline`**

- Loads `scaler.pkl`, `svm_model.pkl`, and `face_landmarker.task` once at import
- `process(frame_bgr)` returns a dict with `prediction`, `label`, `is_drowsy`, `ear`, `ear_diff`, `head_pose`, `head_pose_relative`, `head_off_streak`, `is_head_drowsy`, `calibrating`, `calibration_progress`
- The CLI runner (`scripts/run_realtime.py`) and the Streamlit GUI (`frontend/app.py`) both consume this dict and apply the same smoothing + hysteresis logic

---

## Configuration Reference

All constants live next to the code that uses them and are intentionally **kept in sync** between `src/realtime/pipeline.py`, `scripts/run_realtime.py`, and `frontend/app.py`.

### Eye / image preprocessing — `src/realtime/pipeline.py`

| Constant            | Value      | Purpose                                                |
|---------------------|------------|--------------------------------------------------------|
| `IMG_SIZE`          | `256`      | MobileNetV2 input size (must match training notebook)  |
| `CLAHE_CLIP_LIMIT`  | `2.0`      | CLAHE contrast clip limit                              |
| `CLAHE_TILE_GRID`   | `(8, 8)`   | CLAHE tile grid                                        |
| `EYE_CROP_PADDING`  | `20`       | px around eye landmarks before crop                    |
| `BRIGHTNESS_RANGE`  | `(20, 250)`| reject too dark / too bright frames                    |

### Head-pose channel — `src/realtime/pipeline.py`

| Constant                     | Value | Purpose                                                  |
|------------------------------|-------|----------------------------------------------------------|
| `CALIBRATION_FRAMES`         | `30`  | frames to learn neutral pose (~2 s @ 15 fps)             |
| `HEAD_OFF_AXIS_DELTA_DEG`    | `15.0`| deviation threshold on any axis                          |
| `HEAD_OFF_AXIS_HOLD_FRAMES`  | `22`  | sustained off-axis frames to vote drowsy (~1.5 s)        |

### Smoothing / hysteresis — `frontend/app.py`, `scripts/run_realtime.py`

| Constant            | Value | Purpose                                                  |
|---------------------|-------|----------------------------------------------------------|
| `SMOOTH_WINDOW`     | `30`  | rolling vote window length (~2 s @ 15 fps)               |
| `ENTER_DROWSY_FRAC` | `0.55`| vote ratio required to enter `Drowsy`                    |
| `EXIT_DROWSY_FRAC`  | `0.30`| vote ratio required to leave `Drowsy` (Schmitt trigger)  |

### Audio alerts — `frontend/app.py`

| Constant             | Value | Purpose                          |
|----------------------|-------|----------------------------------|
| `ALERT_COOLDOWN_SEC` | `3.0` | min seconds between audio alerts |

> The alert WAV is generated by `frontend/generate_alert.py` (a 0.7 s two-tone 520/680 Hz sine with attack/release envelope) and saved to `frontend/assets/alert.wav`. If the file is missing, the audio thread synthesizes a fallback tone in memory.

---

## Performance

- **Latency** — ~10-25 Hz on a modern CPU; ~30 Hz with a CUDA GPU
- **Calibration** — ~2 s at the start of every session
- **First-frame to first-decision** — ~1.1 s after calibration finishes (`SMOOTH_WINDOW × ENTER`)
- **Recovery** — ~0.6 s once the driver is clearly awake again

The exact metrics produced by `model/evaluation.ipynb` on the 5,568-sample test set are committed alongside the notebook outputs (open the notebook to inspect the saved confusion matrix, ROC curve, and per-class report).

---

## Troubleshooting

### `TypeError: Descriptors cannot be created directly` on import

You have a newer `protobuf` than TensorFlow 2.10 supports. Re-install the pinned versions:

```bash
pip install --force-reinstall protobuf==3.20.3 mediapipe==0.10.9
```

### `mp.solutions.face_mesh` does not exist

You upgraded `mediapipe` past `0.10.9`. The legacy `solutions.face_mesh` API used by the feature-extraction notebooks was removed in 0.10.21. Either pin to `0.10.9` (recommended) or migrate the notebook to the new `mediapipe.tasks.vision.FaceLandmarker` API (which is what `src/realtime/pipeline.py` already uses).

### `FileNotFoundError: face_landmarker.task`

You did not download the MediaPipe model file. See [Installation › Download the MediaPipe FaceLandmarker model](#download-the-mediapipe-facelandmarker-model).

### `FileNotFoundError: scaler.pkl` / `svm_model.pkl`

Run the training pipeline once:

```bash
jupyter nbconvert --to notebook --execute feature_extraction/feature_extraction.ipynb
jupyter nbconvert --to notebook --execute model/svm_training.ipynb
```

### Webcam cannot be opened

- Make sure no other application is holding the camera
- On Windows, the pipeline tries `CAP_DSHOW` → `CAP_MSMF` → `CAP_ANY` automatically
- Run `python frontend/test_camera.py` in isolation to confirm the OS-level capture works

### Streamlit shows the page but the video stays blank

The video frame uses `<img src="data:image/jpeg;base64,...">` updated in-place inside `st.empty()`. Make sure your browser is not blocking large data URIs and that the camera was actually opened (check the EVENT LOG panel).

### `InconsistentVersionWarning` from scikit-learn

The pickled SVM/scaler were trained on a slightly different sklearn version. The pipeline silences this specific warning class explicitly. If predictions look off, re-train with your current sklearn version.

---

## Tech Stack

| Layer                  | Library                                | Version  |
|------------------------|----------------------------------------|----------|
| Face landmarks         | `mediapipe` (`tasks.vision.FaceLandmarker`) | `0.10.9` |
| Deep features          | `tensorflow.keras.applications.MobileNetV2` | `2.10.0` |
| Image processing       | `opencv-contrib-python`                | `4.10.0` |
| Numerics               | `numpy`                                | `1.26.4` |
| Classifier             | `sklearn.svm.SVC`                      | latest   |
| Scaling                | `sklearn.preprocessing.StandardScaler` | latest   |
| GUI                    | `streamlit`                            | latest   |
| Audio                  | `pygame.mixer`                         | latest   |
| Plotting               | `matplotlib`, `seaborn`                | `3.10.0` |
| Serialization          | `joblib`                               | latest   |

---

## Contributing

PRs and issues are welcome — especially for:

- Replacing MobileNetV2 with a smaller / faster backbone (MobileNetV3, EfficientNet-Lite)
- Adding more sensor channels (yawn detection, gaze direction)
- Cross-platform packaging (PyInstaller / Docker)
- Multi-face support for shared-cab scenarios
- Migrating the feature-extraction notebook to the new `mediapipe.tasks` API to drop the `mediapipe<=0.10.9` pin

To work on the project locally:

```bash
git clone <your-fork-url>
cd Drowsiness-Project
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for the full text. *(Add a `LICENSE` file at the repo root before publishing if one is not already present.)*

---

## Acknowledgments

- **MediaPipe** for the FaceLandmarker model and the canonical 468-point face mesh
- **TensorFlow / Keras** for MobileNetV2 with ImageNet weights
- **scikit-learn** for the SVM and `StandardScaler`
- **OpenCV** for camera I/O, CLAHE, bilateral filtering, and the HUD overlay
- **Streamlit** for the live GUI
- The drowsiness face-image dataset assembled in `datasets/data_acquisition.ipynb` (see `datasets/dataset_split.csv` for the canonical split and `datasets/duplicates_report.csv` for the deduplication audit)

---

<p align="center">
  Built for a Pattern Recognition course project · Real-time, on-device, no cloud calls.
</p>
