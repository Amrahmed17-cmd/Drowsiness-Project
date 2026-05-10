
from __future__ import annotations

import base64
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))   # so `from src.realtime...` works

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    AUDIO_BACKEND = "pygame"
except Exception:
    AUDIO_BACKEND = "none"   # fallback: no audio

st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
ALERT_COOLDOWN_SEC = 3.0    # min seconds between audio alerts
ASSETS_DIR         = Path(__file__).parent / "assets"
ALERT_SOUND        = ASSETS_DIR / "alert.wav"

SMOOTH_WINDOW      = 30     # frames in the rolling vote window
ENTER_DROWSY_FRAC  = 0.55   # vote ratio to flag Drowsy
EXIT_DROWSY_FRAC   = 0.30   # vote ratio to leave Drowsy (hysteresis)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
  --ink:          #07060f;
  --navy:         #0a0d1f;
  --navy2:        #0f1230;
  --purple-deep:  #130720;
  --glass:        rgba(255,255,255,0.04);
  --glass2:       rgba(255,255,255,0.07);
  --glass3:       rgba(255,255,255,0.10);
  --rim:          rgba(150,80,255,0.18);
  --rim2:         rgba(150,80,255,0.38);
  --rim-burg:     rgba(180,30,80,0.28);
  --violet:       #9b5fff;
  --violet-glow:  rgba(155,95,255,0.4);
  --blue:         #4d9fff;
  --blue-glow:    rgba(77,159,255,0.35);
  --crimson:      #c0204a;
  --crimson-glow: rgba(192,32,74,0.45);
  --rose:         #e0547a;
  --rose-glow:    rgba(224,84,122,0.38);
  --green:        #00ffa3;
  --green-glow:   rgba(0,255,163,0.4);
  --amber:        #ffb340;
  --text:         #dde4f5;
  --text2:        #8090b8;
  --text3:        #3a4568;
  --f-hud:        'Rajdhani', sans-serif;
  --f-mono:       'Share Tech Mono', monospace;
  --f-body:       'Exo 2', sans-serif;
}

/* ── Page chrome ─────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--ink) !important;
  font-family: var(--f-body) !important;
  color: var(--text) !important;
}

/* Layered ambient glow — navy / purple / burgundy corners */
[data-testid="stAppViewContainer"]::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 75% 50% at 10% 0%,   rgba(30,10,90,0.55)  0%, transparent 60%),
    radial-gradient(ellipse 55% 45% at 85% 90%,  rgba(120,10,50,0.35) 0%, transparent 55%),
    radial-gradient(ellipse 60% 60% at 50% 50%,  rgba(15,8,40,0.5)    0%, transparent 70%);
}

/* Scanline texture */
[data-testid="stAppViewContainer"]::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 1;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px
  );
}

[data-testid="stMainBlockContainer"] { position: relative; z-index: 2; }
[data-testid="stHeader"]  { background: transparent !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(
    180deg,
    #0d0520 0%, #120830 30%, #1a0814 65%, #0a0d22 100%
  ) !important;
  border-right: 1px solid var(--rim) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
  color: var(--text2) !important;
  font-family: var(--f-mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 1px !important;
}
[data-testid="stSidebarNav"] a {
  color: var(--text2) !important;
  border-radius: 10px !important;
  transition: background 0.2s, color 0.2s !important;
}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-selected="true"] {
  background: linear-gradient(135deg, rgba(155,95,255,0.15), rgba(192,32,74,0.1)) !important;
  color: var(--violet) !important;
  border: 1px solid var(--rim) !important;
  box-shadow: 0 0 16px rgba(155,95,255,0.12) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--rim2); border-radius: 2px; }

/* ── Page title ──────────────────────────────────────────────────────────── */
.sys-title {
  text-align: center;
  font-family: var(--f-hud);
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 700;
  letter-spacing: 4px;
  text-transform: uppercase;
  background: linear-gradient(90deg, var(--blue) 0%, var(--violet) 35%, #d060a0 65%, var(--rose) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: shimmerTitle 5s linear infinite;
  margin-bottom: 4px;
}
@keyframes shimmerTitle {
  from { background-position: 0% center }
  to   { background-position: 200% center }
}
.sys-subtitle {
  text-align: center; font-family: var(--f-mono);
  font-size: 0.72rem; letter-spacing: 3px; color: var(--text3); margin-bottom: 2px;
}
.sys-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--violet), var(--crimson), transparent);
  opacity: 0.4; margin: 12px auto 0; max-width: 600px;
}

/* ── HUD panels ──────────────────────────────────────────────────────────── */
.hud-panel {
  background: linear-gradient(135deg, rgba(10,5,30,0.9), rgba(22,6,16,0.75));
  border: 1px solid var(--rim);
  border-radius: 16px;
  padding: 20px;
  backdrop-filter: blur(20px);
  position: relative; overflow: hidden;
  transition: border-color 0.4s, box-shadow 0.4s;
}
.hud-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(155,95,255,0.5), rgba(192,32,74,0.4), transparent);
}
.hud-panel.alert-active {
  border-color: rgba(192,32,74,0.55);
  box-shadow: 0 0 32px rgba(192,32,74,0.14), inset 0 0 28px rgba(192,32,74,0.05);
  animation: panelPulse 1.3s ease-in-out infinite;
}
@keyframes panelPulse {
  0%,100% { box-shadow: 0 0 20px rgba(192,32,74,0.12) }
  50%     { box-shadow: 0 0 55px rgba(192,32,74,0.35) }
}

/* ── Video feed ──────────────────────────────────────────────────────────── */
.video-wrapper {
  border: 1px solid rgba(155,95,255,0.3);
  border-radius: 12px; overflow: hidden;
  position: relative;
  background: linear-gradient(135deg, #06030f, #120620);
  aspect-ratio: 4/3;
  box-shadow: inset 0 0 40px rgba(155,95,255,0.06);
}
.video-wrapper img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* Corner brackets — violet */
.video-wrapper::before, .video-wrapper::after {
  content: ''; position: absolute; width: 20px; height: 20px;
  border-color: var(--violet); border-style: solid; z-index: 10;
}
.video-wrapper::before { top: 8px; left: 8px; border-width: 2px 0 0 2px; border-radius: 3px 0 0 0; }
.video-wrapper::after  { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; border-radius: 0 0 3px 0; }
.video-overlay-corners::before {
  content: ''; position: absolute; top: 8px; right: 8px; width: 20px; height: 20px;
  border-top: 2px solid var(--violet); border-right: 2px solid var(--violet);
  border-radius: 0 3px 0 0; z-index: 10;
}
.video-overlay-corners::after {
  content: ''; position: absolute; bottom: 8px; left: 8px; width: 20px; height: 20px;
  border-bottom: 2px solid var(--violet); border-left: 2px solid var(--violet);
  border-radius: 0 0 0 3px; z-index: 10;
}

/* REC badge */
.rec-badge {
  position: absolute; top: 12px; right: 12px;
  background: rgba(192,32,74,0.85); color: #fff;
  font-family: var(--f-mono); font-size: 0.65rem; letter-spacing: 2px;
  padding: 3px 10px; border-radius: 999px;
  display: flex; align-items: center; gap: 6px;
  animation: recBlink 1.4s ease-in-out infinite; z-index: 20;
}
.rec-dot { width: 6px; height: 6px; border-radius: 50%; background: #fff; }
@keyframes recBlink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.fps-badge {
  position: absolute; bottom: 12px; left: 12px;
  background: rgba(7,6,15,0.7); font-family: var(--f-mono);
  font-size: 0.65rem; color: var(--violet);
  padding: 3px 10px; border-radius: 6px; border: 1px solid var(--rim); z-index: 20;
}

/* ── Status block ────────────────────────────────────────────────────────── */
.status-block   { text-align: center; padding: 16px 10px; }
.status-eyebrow {
  font-family: var(--f-mono); font-size: 0.62rem;
  letter-spacing: 3px; color: var(--text3); text-transform: uppercase; margin-bottom: 10px;
}
.status-label {
  font-family: var(--f-hud); font-size: 2.8rem; font-weight: 700;
  letter-spacing: 6px; text-transform: uppercase;
  transition: color 0.4s, text-shadow 0.4s; line-height: 1;
}
.status-awake {
  color: var(--green);
  text-shadow: 0 0 20px var(--green-glow), 0 0 50px rgba(0,255,163,0.2);
}
.status-drowsy {
  color: var(--rose);
  text-shadow: 0 0 20px var(--rose-glow), 0 0 50px rgba(192,32,74,0.3);
  animation: drowsyFlicker 0.8s ease-in-out infinite;
}
.status-idle   { color: var(--text2); text-shadow: none; }
@keyframes drowsyFlicker { 0%,100%{opacity:1} 45%{opacity:0.72} }

/* ── Score bar ───────────────────────────────────────────────────────────── */
.score-label-row {
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--f-mono); font-size: 0.72rem; color: var(--text2); margin-bottom: 8px;
}
.score-value { font-family: var(--f-hud); font-weight: 700; font-size: 1.1rem; color: var(--violet); }
.score-bar-bg {
  height: 10px; border-radius: 999px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--rim); overflow: hidden; position: relative;
}
.score-bar-fill {
  height: 100%; border-radius: 999px;
  transition: width 0.5s cubic-bezier(.22,1,.36,1), background 0.5s; position: relative;
}
.score-bar-fill::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 60%, rgba(255,255,255,0.25));
  border-radius: 999px;
}
.score-ticks {
  display: flex; justify-content: space-between;
  font-family: var(--f-mono); font-size: 0.58rem; color: var(--text3);
  margin-top: 5px; padding: 0 2px;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 10px; margin-top: 16px; }
.metric-card {
  flex: 1;
  background: linear-gradient(135deg, rgba(13,5,32,0.9), rgba(26,8,20,0.7));
  border: 1px solid var(--rim); border-radius: 10px; padding: 12px 10px; text-align: center;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
  border-color: rgba(155,95,255,0.4);
  box-shadow: 0 0 20px rgba(155,95,255,0.1);
}
.metric-card-label {
  font-family: var(--f-mono); font-size: 0.60rem; letter-spacing: 2px;
  color: var(--text3); text-transform: uppercase; margin-bottom: 5px;
}
.metric-card-value { font-family: var(--f-hud); font-size: 1.35rem; font-weight: 700; color: var(--violet); }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--f-hud) !important; font-size: 1rem !important;
  font-weight: 700 !important; letter-spacing: 3px !important;
  text-transform: uppercase !important; border-radius: 10px !important;
  padding: 14px 28px !important; width: 100% !important; border: none !important;
  transition: transform 0.18s, box-shadow 0.18s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* START — violet/purple */
div[data-testid="column"]:nth-child(1) .stButton > button {
  background: linear-gradient(135deg, #200a50, #5a1fa0, #9b5fff) !important;
  color: #fff !important;
  box-shadow: 0 0 20px rgba(155,95,255,0.3), 0 4px 14px rgba(0,0,0,0.5) !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
  box-shadow: 0 0 36px rgba(155,95,255,0.55), 0 6px 20px rgba(0,0,0,0.5) !important;
}

/* STOP — burgundy/crimson */
div[data-testid="column"]:nth-child(2) .stButton > button {
  background: linear-gradient(135deg, #1a0510, #6b0020, #c0204a) !important;
  color: #fff !important;
  box-shadow: 0 0 20px rgba(192,32,74,0.3), 0 4px 14px rgba(0,0,0,0.5) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
  box-shadow: 0 0 36px rgba(192,32,74,0.55), 0 6px 20px rgba(0,0,0,0.5) !important;
}

/* ── Text inputs / textareas / selects ───────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
  background: rgba(10,5,28,0.85) !important;
  border: 1px solid var(--rim) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--f-body) !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: rgba(155,95,255,0.55) !important;
  box-shadow: 0 0 0 3px rgba(155,95,255,0.12), 0 0 18px rgba(155,95,255,0.1) !important;
  outline: none !important;
}

/* ── Log console ─────────────────────────────────────────────────────────── */
.log-console {
  background: rgba(5,3,18,0.75);
  border: 1px solid var(--rim); border-radius: 10px;
  padding: 12px 14px; font-family: var(--f-mono); font-size: 0.72rem; color: var(--text2);
  height: 120px; overflow-y: auto; line-height: 1.9;
}
.log-entry-info  { color: var(--blue);  }
.log-entry-warn  { color: var(--amber); }
.log-entry-alert { color: var(--rose);  }
.log-entry-ok    { color: var(--green); }

/* ── Section header ──────────────────────────────────────────────────────── */
.sec-header {
  font-family: var(--f-mono); font-size: 0.65rem; letter-spacing: 3px; color: var(--text3);
  text-transform: uppercase; border-left: 2px solid var(--violet);
  padding-left: 10px; margin-bottom: 14px;
}
</style>
"""

def _init_state():
    defaults = {
        "running":         False,
        "cap":             None,
        "status":          "Idle",
        "fatigue_score":   0,
        "fps":             0.0,
        "frames_total":    0,
        "alerts_total":    0,
        "session_seconds": 0,
        "log_lines":       [],
        "start_time":      None,
        "last_alert_time": 0.0,
        "fps_counter":     0,
        "fps_t0":          0.0,
        "session_t0":      0.0,
        "drowsy_buffer":      deque(maxlen=SMOOTH_WINDOW),
        "last_ear":           None,
        "last_ear_diff":      None,
        "last_raw_label":     None,
        "last_head_pose":     None,
        "last_head_pose_rel": None,
        "head_off_streak":    0,
        "is_head_drowsy":     False,
        "calibrating":        False,
        "calibration_progress": 0.0,
        "no_face_streak":     0,
        "prev_status":        "Idle",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

_PIPELINE_LOAD_ERROR: Optional[str] = None


@st.cache_resource(   # load model once across reruns
    show_spinner="Loading drowsiness pipeline (MediaPipe + MobileNetV2 + SVM)…",
)
def get_pipeline():
    global _PIPELINE_LOAD_ERROR
    try:
        from src.realtime.pipeline import DrowsinessPipeline
        _PIPELINE_LOAD_ERROR = None
        return DrowsinessPipeline()
    except Exception as exc:
        _PIPELINE_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _real_pipeline(frame: np.ndarray) -> tuple[str, int]:
    pipeline = get_pipeline()
    if pipeline is None:
        return "Error", 0

    result = pipeline.process(frame)            # eye + head-pose inference
    buf    = st.session_state.drowsy_buffer

    if result is None:
        st.session_state.no_face_streak += 1
        st.session_state.last_raw_label  = None
        if not buf:
            return "No Face", 0
    else:
        st.session_state.no_face_streak       = 0
        st.session_state.last_ear             = result["ear"]
        st.session_state.last_ear_diff        = result["ear_diff"]
        st.session_state.last_raw_label       = result["label"]
        st.session_state.last_head_pose       = result["head_pose"]
        st.session_state.last_head_pose_rel   = result["head_pose_relative"]
        st.session_state.head_off_streak      = result["head_off_streak"]
        st.session_state.is_head_drowsy       = result["is_head_drowsy"]
        st.session_state.calibrating          = result["calibrating"]
        st.session_state.calibration_progress = result["calibration_progress"]

        if result["calibrating"]:
            return "Calibrating", 0    # don't vote until baseline locked in

        frame_drowsy = bool(result["is_drowsy"]) or bool(result["is_head_drowsy"])  # eye OR head
        buf.append(1 if frame_drowsy else 0)

    drowsy_score = sum(buf) / len(buf) if buf else 0.0
    score_int    = int(round(drowsy_score * 100))   # 0-100 fatigue score

    prev_drowsy = (st.session_state.prev_status == "Drowsy")
    if prev_drowsy:
        is_drowsy = drowsy_score >= EXIT_DROWSY_FRAC    # easier to stay
    else:
        is_drowsy = drowsy_score >= ENTER_DROWSY_FRAC   # harder to enter

    return ("Drowsy" if is_drowsy else "Awake"), score_int

def _play_alert():
    now = time.time()
    if now - st.session_state.last_alert_time < ALERT_COOLDOWN_SEC:
        return    # cooldown to prevent audio spam
    st.session_state.last_alert_time = now
    if AUDIO_BACKEND != "pygame":
        return
    try:
        import threading
        def _play():
            try:
                if ALERT_SOUND.exists():
                    import pygame
                    pygame.mixer.Sound(str(ALERT_SOUND)).play()
                else:
                    import pygame, numpy as _np
                    sr, freq, dur = 44100, 520, 0.6
                    t    = _np.linspace(0, dur, int(sr * dur), False)
                    wave = (_np.sin(2 * _np.pi * freq * t) * 32767).astype(_np.int16)
                    pygame.sndarray.make_sound(_np.column_stack([wave, wave])).play()
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()
    except Exception:
        pass

def _log(msg: str, level: str = "info"):
    ts     = time.strftime("%H:%M:%S")
    prefix = {"info": "ℹ", "ok": "✓", "warn": "⚠", "alert": "⚡"}.get(level, "•")
    st.session_state.log_lines.append((f"[{ts}]  {prefix}  {msg}", level))

def _open_camera() -> bool:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    cap = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):   # try several backends
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
            cap.release()
            cap = None
        except Exception:
            cap = None

    if cap is None or not cap.isOpened():
        _log("Cannot open webcam — check permissions & that no other app is using it", "alert")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always grab the latest frame
    st.session_state.cap = cap
    _log("Webcam opened successfully", "ok")
    return True


def _close_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    _log("Webcam released — session ended", "info")

def start_detection():
    if st.session_state.running:
        return
    st.session_state.frames_total         = 0
    st.session_state.alerts_total         = 0
    st.session_state.session_seconds      = 0
    st.session_state.log_lines            = []
    st.session_state.fps                  = 0.0
    st.session_state.fps_counter          = 0
    st.session_state.fps_t0               = time.perf_counter()
    st.session_state.session_t0           = time.time()
    st.session_state.start_time           = time.time()
    st.session_state.drowsy_buffer        = deque(maxlen=SMOOTH_WINDOW)
    st.session_state.last_ear             = None
    st.session_state.last_ear_diff        = None
    st.session_state.last_raw_label       = None
    st.session_state.last_head_pose       = None
    st.session_state.last_head_pose_rel   = None
    st.session_state.head_off_streak      = 0
    st.session_state.is_head_drowsy       = False
    st.session_state.calibrating          = True
    st.session_state.calibration_progress = 0.0
    st.session_state.no_face_streak       = 0
    st.session_state.prev_status          = "Idle"

    pipeline = get_pipeline()
    if pipeline is None:
        _log(
            f"Pipeline failed to load: {_PIPELINE_LOAD_ERROR}",
            "alert",
        )
        return

    pipeline.reset_calibration()    # fresh head-pose baseline per session

    if not _open_camera():
        return

    st.session_state.running = True
    st.session_state.status  = "Calibrating"
    _log("Session started — calibrating neutral head pose…", "info")


def stop_detection():
    _close_camera()
    st.session_state.running       = False
    st.session_state.status        = "Idle"
    st.session_state.fatigue_score = 0
    st.session_state.fps           = 0.0
    _log("Detection stopped by user", "warn")

def _grab_frame() -> Optional[str]:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        _log("Camera handle lost — stopping", "alert")
        stop_detection()
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        _log("Frame read failed — retrying next cycle", "warn")
        return None

    try:
        status, score = _real_pipeline(frame)   # run inference
    except Exception as exc:
        status, score = "Error", 0
        _log(f"Pipeline exception: {exc}", "alert")

    prev_status = st.session_state.prev_status
    st.session_state.status          = status
    st.session_state.fatigue_score   = score
    st.session_state.frames_total   += 1
    st.session_state.session_seconds = int(time.time() - st.session_state.session_t0)

    if prev_status == "Calibrating" and status != "Calibrating":
        _log("Calibration complete — monitoring active", "ok")

    if status == "Drowsy" and prev_status not in ("Drowsy", "Calibrating"):
        st.session_state.alerts_total += 1   # only count fresh transitions
        head_note = " (head + eyes)" if st.session_state.is_head_drowsy else ""
        _log(f"Drowsiness detected (score {score}){head_note}", "alert")

    if status == "Drowsy":
        _play_alert()

    st.session_state.prev_status = status

    st.session_state.fps_counter += 1
    elapsed = time.perf_counter() - st.session_state.fps_t0
    if elapsed >= 1.0:                                # update FPS every second
        st.session_state.fps         = round(st.session_state.fps_counter / elapsed, 1)
        st.session_state.fps_counter = 0
        st.session_state.fps_t0      = time.perf_counter()

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # encode for HTML
    return base64.b64encode(buf.tobytes()).decode()

def _score_bar_css(score: int) -> str:
    if score < 35:
        return f"width:{score}%;background:linear-gradient(90deg,#00c97a,#00ffa3);"
    if score < 60:
        return (
            f"width:{score}%;"
            f"background:linear-gradient(90deg,#6b0030,#c0204a,#e0547a);"
            f"box-shadow:0 0 8px rgba(192,32,74,0.4);"
        )
    return (
        f"width:{score}%;"
        f"background:linear-gradient(90deg,#8c0030,#c0204a,#e0547a);"
        f"box-shadow:0 0 14px rgba(224,84,122,0.6);"
    )


def _fmt_seconds(s: int) -> str:
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"


def _placeholder_html() -> str:
    return """
    <div style="width:100%;aspect-ratio:4/3;
                background:linear-gradient(135deg,#06030f,#120620);
                border:1px solid rgba(155,95,255,0.2);border-radius:12px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;
                box-shadow:inset 0 0 40px rgba(155,95,255,0.06);">
        <div style="font-size:3.5rem;opacity:0.2;">📷</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                    letter-spacing:3px;color:rgba(90,70,140,0.55);">CAMERA OFFLINE</div>
    </div>"""


def _video_html(frame_b64: Optional[str], status: str, fps: float) -> str:
    if not frame_b64:
        return _placeholder_html()
    is_drowsy    = status == "Drowsy"
    border_color = "rgba(192,32,74,0.6)" if is_drowsy else "rgba(155,95,255,0.3)"
    pulse_cls    = "alert-active" if is_drowsy else ""
    return f"""
    <div class="hud-panel {pulse_cls}" style="padding:0;border-color:{border_color};">
        <div class="video-wrapper video-overlay-corners">
            <img src="data:image/jpeg;base64,{frame_b64}" />
            <div class="rec-badge"><div class="rec-dot"></div> REC</div>
            <div class="fps-badge">{fps} FPS</div>
        </div>
    </div>"""


def _status_html(status: str) -> str:
    is_drowsy      = status == "Drowsy"
    is_awake       = status == "Awake"
    is_calibrating = status == "Calibrating"
    status_css     = "status-drowsy" if is_drowsy else "status-awake" if is_awake else "status-idle"
    status_icon    = "⚠" if is_drowsy else "✓" if is_awake else "◉" if is_calibrating else "○"
    pulse_cls      = "alert-active" if is_drowsy else ""

    extra = ""
    if is_calibrating:
        pct = int(round(st.session_state.calibration_progress * 100))
        extra = (
            '<div style="margin-top:14px;">'
              '<div class="score-bar-bg">'
                f'<div class="score-bar-fill" style="width:{pct}%;'
                'background:linear-gradient(90deg,#4d9fff,#9b5fff);'
                'box-shadow:0 0 12px rgba(155,95,255,0.4);"></div>'
              '</div>'
              '<div style="font-family:var(--f-mono);font-size:0.62rem;'
              'letter-spacing:2px;color:var(--text3);margin-top:6px;'
              f'text-align:center;">CALIBRATING NEUTRAL POSE · {pct}%</div>'
            '</div>'
        )

    return (
        f'<div class="hud-panel {pulse_cls}">'
          '<div class="status-block">'
            '<div class="status-eyebrow">DETECTION STATUS</div>'
            f'<div class="status-label {status_css}">{status_icon} {status.upper()}</div>'
            f'{extra}'
          '</div>'
        '</div>'
    )


def _score_html(score: int) -> str:
    return f"""
    <div class="hud-panel">
        <div class="score-label-row">
            <span>FATIGUE LEVEL</span>
            <span class="score-value">{score}<span style="font-size:0.75rem;color:var(--text2);">/100</span></span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="{_score_bar_css(score)}"></div>
        </div>
        <div class="score-ticks">
            <span>ALERT</span><span>MODERATE</span><span>FATIGUED</span><span>CRITICAL</span>
        </div>
    </div>"""


def _metrics_html() -> str:
    dur         = _fmt_seconds(st.session_state.session_seconds)
    alert_color = "var(--rose)" if st.session_state.alerts_total > 0 else "var(--violet)"
    return f"""
    <div class="hud-panel">
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-card-label">Duration</div>
                <div class="metric-card-value">{dur}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Frames</div>
                <div class="metric-card-value">{st.session_state.frames_total:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Alerts</div>
                <div class="metric-card-value" style="color:{alert_color};">{st.session_state.alerts_total}</div>
            </div>
        </div>
    </div>"""


def _system_html() -> str:
    audio_status  = f"✓ {AUDIO_BACKEND}" if AUDIO_BACKEND != "none" else "✗ No audio"
    running_str   = "● ACTIVE" if st.session_state.running else "○ STANDBY"
    running_color = "var(--green)" if st.session_state.running else "var(--text3)"

    if _PIPELINE_LOAD_ERROR:
        model_str, model_color = "✗ FAILED", "var(--rose)"
    else:
        model_str, model_color = "✓ READY", "var(--green)"

    ear_str = (
        f"{st.session_state.last_ear:.3f}"
        if st.session_state.last_ear is not None else "—"
    )
    ear_diff_str = (
        f"{st.session_state.last_ear_diff:.3f}"
        if st.session_state.last_ear_diff is not None else "—"
    )
    raw_label = st.session_state.last_raw_label or "—"

    rel = st.session_state.last_head_pose_rel
    if rel is not None:
        pitch_str = f"{rel[0]:+5.1f}°"
        yaw_str   = f"{rel[1]:+5.1f}°"
        roll_str  = f"{rel[2]:+5.1f}°"
    else:
        pitch_str = yaw_str = roll_str = "—"

    if st.session_state.calibrating:
        head_status, head_color = "CALIBRATING", "var(--blue)"
    elif st.session_state.is_head_drowsy:
        head_status, head_color = "OFF-AXIS", "var(--rose)"
    elif st.session_state.head_off_streak > 0:
        head_status, head_color = "DRIFTING", "var(--amber)"
    else:
        head_status, head_color = "ALIGNED", "var(--green)"

    return f"""
    <div class="hud-panel">
        <div style="font-family:var(--f-mono);font-size:0.72rem;color:var(--text2);line-height:2.2;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">ENGINE</span>
                <span style="color:{running_color};font-weight:700;">{running_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">MODEL</span>
                <span style="color:{model_color};font-weight:700;">{model_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">EYE LABEL</span>
                <span style="color:var(--violet);">{raw_label}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">EAR</span>
                <span style="color:var(--violet);">{ear_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">EAR DIFF</span>
                <span style="color:var(--violet);">{ear_diff_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">HEAD</span>
                <span style="color:{head_color};font-weight:700;">{head_status}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">PITCH / YAW / ROLL</span>
                <span style="color:var(--violet);">{pitch_str} · {yaw_str} · {roll_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">FPS</span>
                <span style="color:var(--violet);">{st.session_state.fps}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">AUDIO</span>
                <span style="color:var(--green);">{audio_status}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text3);">RESOLUTION</span>
                <span>{FRAME_WIDTH}×{FRAME_HEIGHT}</span>
            </div>
        </div>
    </div>"""


def _log_html() -> str:
    body = "".join(
        f'<div class="log-entry-{lvl}">{ln}</div>'
        for ln, lvl in reversed(st.session_state.log_lines[-20:])
    ) or '<div style="color:var(--text3);">-- No events yet --</div>'
    return f'<div class="log-console">{body}</div>'

def render():
    st.markdown(CSS, unsafe_allow_html=True)

    if "pipeline_warmed" not in st.session_state:   # eager-load on first render
        get_pipeline()
        st.session_state.pipeline_warmed = True
        if _PIPELINE_LOAD_ERROR:
            _log(f"Model load failed: {_PIPELINE_LOAD_ERROR}", "alert")
        else:
            _log("Drowsiness model loaded — ready", "ok")

    st.markdown("""
    <div style="padding:28px 0 20px;">
        <div class="sys-title">Real-Time Drowsiness Detection System</div>
        <div class="sys-subtitle">DRIVER SAFETY MONITORING  ·  AI-POWERED  ·  LIVE ANALYSIS</div>
        <div class="sys-divider"></div>
    </div>""", unsafe_allow_html=True)

    col_video, col_dash = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown('<div class="sec-header">LIVE FEED</div>', unsafe_allow_html=True)
        video_slot = st.empty()

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2, gap="medium")
        with b1:
            st.button(
                "▶  START",
                disabled=st.session_state.running,
                use_container_width=True,
                on_click=start_detection,
                key="btn_start",
            )
        with b2:
            st.button(
                "■  STOP",
                disabled=not st.session_state.running,
                use_container_width=True,
                on_click=stop_detection,
                key="btn_stop",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">EVENT LOG</div>', unsafe_allow_html=True)
        log_slot = st.empty()

    with col_dash:
        st.markdown('<div class="sec-header">DRIVER STATUS</div>', unsafe_allow_html=True)
        status_slot = st.empty()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">FATIGUE SCORE</div>', unsafe_allow_html=True)
        score_slot = st.empty()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">SESSION METRICS</div>', unsafe_allow_html=True)
        metrics_slot = st.empty()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">SYSTEM</div>', unsafe_allow_html=True)
        system_slot = st.empty()

    video_slot.markdown(
        _video_html(None, st.session_state.status, st.session_state.fps),
        unsafe_allow_html=True,
    )
    status_slot.markdown(_status_html(st.session_state.status), unsafe_allow_html=True)
    score_slot.markdown(_score_html(st.session_state.fatigue_score), unsafe_allow_html=True)
    metrics_slot.markdown(_metrics_html(), unsafe_allow_html=True)
    system_slot.markdown(_system_html(), unsafe_allow_html=True)
    log_slot.markdown(_log_html(), unsafe_allow_html=True)

    if st.session_state.running:        # live loop owns the session
        last_metrics_t = 0.0
        last_system_t  = 0.0
        last_log_count = -1

        while st.session_state.running:
            frame_b64 = _grab_frame()    # one frame -> inference -> b64
            now       = time.time()

            video_slot.markdown(
                _video_html(frame_b64, st.session_state.status, st.session_state.fps),
                unsafe_allow_html=True,
            )
            status_slot.markdown(
                _status_html(st.session_state.status), unsafe_allow_html=True
            )
            score_slot.markdown(
                _score_html(st.session_state.fatigue_score), unsafe_allow_html=True
            )

            if now - last_metrics_t > 0.25:    # ~4 Hz
                metrics_slot.markdown(_metrics_html(), unsafe_allow_html=True)
                last_metrics_t = now

            if now - last_system_t > 0.5:      # ~2 Hz
                system_slot.markdown(_system_html(), unsafe_allow_html=True)
                last_system_t = now

            log_count = len(st.session_state.log_lines)
            if log_count != last_log_count:    # only redraw on change
                log_slot.markdown(_log_html(), unsafe_allow_html=True)
                last_log_count = log_count

            time.sleep(0.005)   # yield to event loop


render()
