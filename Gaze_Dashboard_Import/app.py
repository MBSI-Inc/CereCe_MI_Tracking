"""
Cerebruh Gaze Dashboard — FastAPI + WebSocket
==============================================
Run:  python app.py
Then open http://127.0.0.1:8050

Replaces the old Dash-based dashboard with a modern
FastAPI backend streaming real-time data over WebSocket.
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from statistics import median, mean

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── GazeTracker import ───────────────────────────────────────────────────────
IMPORT_ERROR_MSG = ""
try:
    from facemesh_gazetracker import GazeTracker
    DEMO_MODE = False
    print("[+] GazeTracker imported.")
except Exception:
    DEMO_MODE = True
    IMPORT_ERROR_MSG = traceback.format_exc()
    print("[!] GazeTracker not found - DEMO mode.\n", IMPORT_ERROR_MSG)

# ── Calibration utilities import ─────────────────────────────────────────────
try:
    from util import Util
    print("[+] Util imported.")
except Exception:
    print("[!] Util import failed.\n", traceback.format_exc())

# ── Unity networking (legacy UDP path — removed in Phase 4) ─────────────────
# The old socket-based integration is superseded by the in-browser WebGL bridge
# (see static/index.html wsGaze.onmessage -> unityInstance.SendMessage).
# UNITY_AVAILABLE is kept at False so legacy /api/state consumers don't crash.
UNITY_AVAILABLE = False


# ── Camera init with Windows-safe fallback ───────────────────────────────────
def _open_camera_with_fallback(index: int = 0):
    """
    Open a webcam robustly on Windows.

    Pattern: try CAP_DSHOW, CAP_MSMF, then default — for indices 0, 1, 2.
    Verifies each candidate with an actual read() call because
    cv2.VideoCapture on Windows can return an object where isOpened() is True
    but all read() calls return (False, None) when the device is busy or the
    DirectShow graph failed to fully initialize.

    Returns: cv2.VideoCapture instance (already opened and verified to deliver
             at least one frame).

    Raises: RuntimeError with a human-readable message suggesting Windows
            Privacy Settings / closing other camera apps.
    """
    import sys as _sys
    attempts = []
    backends = (
        [(cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_MSMF, "CAP_MSMF"), (None, "default")]
        if _sys.platform == "win32"
        else [(None, "default")]
    )
    indices_to_try = [index, index + 1, index + 2] if index == 0 else [index]

    for idx in indices_to_try:
        for backend, name in backends:
            try:
                cam = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            except Exception as e:
                attempts.append(f"index={idx} backend={name}: exception {e!r}")
                continue
            opened = cam.isOpened()
            if not opened:
                attempts.append(f"index={idx} backend={name}: isOpened()=False")
                try: cam.release()
                except Exception: pass
                continue
            # Real test: does read() actually deliver a frame?
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"[cam] opened index={idx} backend={name} frame={frame.shape}")
                return cam
            attempts.append(f"index={idx} backend={name}: isOpened()=True but read() returned ret={ret} frame={'None' if frame is None else 'valid'}")
            try: cam.release()
            except Exception: pass

    diag = "; ".join(attempts) if attempts else "no backends attempted"
    raise RuntimeError(
        f"Camera init failed. Tried: {diag}. "
        "Check Windows Settings -> Privacy -> Camera -> allow desktop apps. "
        "Close other apps that might be using your camera (Zoom, Teams, OBS, browser tabs)."
    )


# ── Demo tracker ─────────────────────────────────────────────────────────────
class DemoTracker:
    def __init__(self):
        self.running = False
        self._out = None
        self._t = 0.0
        self._lock = threading.Lock()
        self.config = {
            "center_offset": 0.0,
            "head_rotation_center_offset": 0.0,
            "left_threshold": 0.35,
            "right_threshold": 0.65,
            "cruise_control_enabled": False,
            "controller_mode": 0,
        }
        self.value_buffer = {
            "gaze_horizontal_full": [0.5],
            "head_rotation": [0.0],
        }

    def _loop(self):
        while self.running:
            self._t += 0.08
            h = int(np.sin(self._t * 0.7) * 75)
            v = int(20 + np.sin(self._t * 0.3) * 18)
            with self._lock:
                self._out = {
                    "combined_gaze_horizontal": h,
                    "combined_gaze_vertical": v,
                    "blinked": (int(self._t * 12) % 45 == 0),
                    "face_detected": True,
                    "move_enabled": True,
                    "_yaw": round(np.sin(self._t * 0.5) * 12, 2),
                    "_pitch": round(np.cos(self._t * 0.4) * 8, 2),
                    "_roll": round(np.sin(self._t * 0.2) * 5, 2),
                }
            time.sleep(0.08)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def get_gaze_async(self):
        with self._lock:
            return dict(self._out) if self._out else None

    def get_frame(self):
        h, w = 360, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            frame[i] = [int(15 + i * 0.04), int(8 + i * 0.02), int(35 + i * 0.08)]
        cx, cy = w // 2, h // 2
        ex = int(cx + np.sin(self._t * 0.7) * 22)
        cv2.circle(frame, (cx, cy), 90, (50, 160, 110), 2)
        for dx in (-30, 30):
            cv2.circle(frame, (ex + dx, cy - 10), 8, (90, 200, 160), -1)
            cv2.circle(frame, (ex + dx, cy - 10), 3, (20, 70, 50), -1)
        cv2.putText(frame, "DEMO", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
        return frame


# ── Application state ────────────────────────────────────────────────────────
tracker = None
MAX_HISTORY = 80
gaze_history = deque(maxlen=MAX_HISTORY)

# Camera frame buffers
_webcam_b64 = ""
_unity_b64 = ""
_last_webcam = None
_buf_lock = threading.Lock()
unity_conn = False
_show_unity_feed = False

# Dashboard state
state = {
    "cal_step": 0,       # 0=idle, 1=calibrating, 4=done
    "cal_done": False,
    "running": False,
    "unity_feed": False,
    "controller_mode": 0,
    "cruise_control": False,
    "left_threshold": 0.35,
    "right_threshold": 0.65,
}

# Calibration thread state
_cal_thread = None
_cal_stop = threading.Event()
cal_progress = {
    "active": False,
    "step_name": "",
    "step_index": 0,     # 0-5 (center, left, right, l-blink, r-blink, done)
    "total_steps": 5,
    "time_left": 0.0,
    "total_elapsed": 0.0,
    "total_duration": 22.0,
    "done": False,
    "error": "",
}

# ── Calibration constants (matching calibrate.py) ────────────────────────────
CAL_DURATION_EACH_STEP = 3
CAL_GET_READY_TIME = 2
CAL_DIRECTION_LEFT = 0.1
CAL_DIRECTION_CENTER = 0.5
CAL_DIRECTION_RIGHT = 0.9


def _calibration_worker():
    """Runs calibrate.py logic in a background thread, writing frames to _webcam_b64."""
    global tracker, _webcam_b64, _last_webcam

    cal_progress.update({
        "active": True, "step_name": "get_ready", "step_index": 0,
        "time_left": CAL_GET_READY_TIME, "total_elapsed": 0.0,
        "total_duration": CAL_GET_READY_TIME * 3 + CAL_DURATION_EACH_STEP * 5 + 1,
        "done": False, "error": "",
    })

    cam = None
    face_mesh = None
    try:
        cam = _open_camera_with_fallback(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        ema_alpha = 0.3
        ema_left = 0.0
        ema_right = 0.0

        # Load config
        cfg_path = str(_HERE.parent / "config.json")
        with open(cfg_path, "r") as f:
            data = json.load(f)

        center_offset_data = []
        left_threshold_data = []
        right_threshold_data = []
        left_closed_data = []
        right_closed_data = []

        start_time = time.time()
        GR = CAL_GET_READY_TIME
        DUR = CAL_DURATION_EACH_STEP
        total_dur = GR * 3 + DUR * 5 + 1

        while not _cal_stop.is_set():
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks

            elapsed = time.time() - start_time
            cal_progress["total_elapsed"] = elapsed
            cal_progress["total_duration"] = total_dur

            # Determine current step for progress reporting
            if elapsed < GR:
                cal_progress["step_name"] = "get_ready"
                cal_progress["step_index"] = 0
                cal_progress["time_left"] = GR - elapsed
                cv2.putText(frame, "Get ready - look at the green dot",
                            (frame_w // 2 - 280, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            elif elapsed < GR + DUR:
                cal_progress["step_name"] = "center"
                cal_progress["step_index"] = 1
                cal_progress["time_left"] = GR + DUR - elapsed
            elif elapsed < GR + DUR * 2:
                cal_progress["step_name"] = "left"
                cal_progress["step_index"] = 2
                cal_progress["time_left"] = GR + DUR * 2 - elapsed
            elif elapsed < GR + DUR * 3:
                cal_progress["step_name"] = "right"
                cal_progress["step_index"] = 3
                cal_progress["time_left"] = GR + DUR * 3 - elapsed
            elif elapsed < GR * 2 + DUR * 3:
                cal_progress["step_name"] = "prep_left_blink"
                cal_progress["step_index"] = 3
                cal_progress["time_left"] = GR * 2 + DUR * 3 - elapsed
            elif elapsed < GR * 2 + DUR * 4:
                cal_progress["step_name"] = "left_blink"
                cal_progress["step_index"] = 4
                cal_progress["time_left"] = GR * 2 + DUR * 4 - elapsed
            elif elapsed < GR * 3 + DUR * 4:
                cal_progress["step_name"] = "prep_right_blink"
                cal_progress["step_index"] = 4
                cal_progress["time_left"] = GR * 3 + DUR * 4 - elapsed
            elif elapsed < GR * 3 + DUR * 5:
                cal_progress["step_name"] = "right_blink"
                cal_progress["step_index"] = 5
                cal_progress["time_left"] = GR * 3 + DUR * 5 - elapsed

            if landmark_points:
                landmarks = landmark_points[0].landmark

                # Get eye direction ratios with EMA smoothing
                frame, right_dir, _ = Util.get_eye_direction_ratio(frame, landmarks, True)
                ema_right = ema_alpha * right_dir + (1 - ema_alpha) * ema_right
                right_dir = ema_right

                frame, left_dir, _ = Util.get_eye_direction_ratio(frame, landmarks, False)
                ema_left = ema_alpha * left_dir + (1 - ema_alpha) * ema_left
                left_dir = ema_left

                direction = (left_dir + right_dir) / 2

                # ── Center step ──
                if GR < elapsed < GR + DUR:
                    center = (round(frame_w * CAL_DIRECTION_CENTER), round(frame_h * 0.4))
                    radius = max(1, round(15 * (1 - (elapsed - GR) / DUR)))
                    cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    cv2.putText(frame, "Look at the green dot",
                                (frame_w // 2 - 180, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    center_offset_data.append(-direction)
                    data["center_offset"] = median(center_offset_data)

                # ── Left step ──
                if GR + DUR < elapsed < GR + DUR * 2:
                    center = (round(frame_w * CAL_DIRECTION_LEFT), round(frame_h * 0.4))
                    radius = max(1, round(15 * (1 - (elapsed - GR - DUR) / DUR)))
                    cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    cv2.putText(frame, "Look at the green dot",
                                (frame_w // 2 - 180, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    left_threshold_data.append(data["center_offset"] + direction)
                    data["left_threshold"] = median(left_threshold_data)

                # ── Right step ──
                if GR + DUR * 2 < elapsed < GR + DUR * 3:
                    center = (round(frame_w * CAL_DIRECTION_RIGHT), round(frame_h * 0.4))
                    radius = max(1, round(15 * (1 - (elapsed - GR - DUR * 2) / DUR)))
                    cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    cv2.putText(frame, "Look at the green dot",
                                (frame_w // 2 - 180, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    right_threshold_data.append(data["center_offset"] + direction)
                    data["right_threshold"] = median(right_threshold_data)

                # ── Instruction: close left eye ──
                if GR + DUR * 3 < elapsed < GR * 2 + DUR * 3:
                    cv2.putText(frame, "Close LEFT eye, open right eye",
                                (frame_w // 2 - 260, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

                # ── Left eye closed step ──
                if GR * 2 + DUR * 3 < elapsed < GR * 2 + DUR * 4:
                    center = (round(frame_w * CAL_DIRECTION_CENTER), round(frame_h * 0.4))
                    radius = max(1, round(15 * (1 - (elapsed - GR * 2 - DUR * 3) / DUR)))
                    cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    cv2.putText(frame, "Close LEFT eye, open right eye",
                                (frame_w // 2 - 260, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                    ratio, _, _ = Util.calculate_eye_slit_h_nosebridge_ratio(
                        frame, landmarks, right_eye=False, debug=True, flip=True)
                    left_closed_data.append(round(ratio, 2))

                # ── Instruction: close right eye ──
                if GR * 2 + DUR * 4 < elapsed < GR * 3 + DUR * 4:
                    cv2.putText(frame, "Close RIGHT eye, open left eye",
                                (frame_w // 2 - 260, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

                # ── Right eye closed step ──
                if GR * 3 + DUR * 4 < elapsed < GR * 3 + DUR * 5:
                    center = (round(frame_w * CAL_DIRECTION_CENTER), round(frame_h * 0.4))
                    radius = max(1, round(15 * (1 - (elapsed - GR * 3 - DUR * 4) / DUR)))
                    cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    cv2.putText(frame, "Close RIGHT eye, open left eye",
                                (frame_w // 2 - 260, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                    ratio, _, _ = Util.calculate_eye_slit_h_nosebridge_ratio(
                        frame, landmarks, right_eye=True, debug=True, flip=True)
                    right_closed_data.append(round(ratio, 2))

                # ── Done ──
                if elapsed > GR * 3 + DUR * 5 + 1:
                    if left_closed_data:
                        data["left_eye_closed_threshold"] = median(left_closed_data)
                        print(f"[cal] Left blink range: {left_closed_data[0]}-{left_closed_data[-1]}, median: {median(left_closed_data)}")
                    if right_closed_data:
                        data["right_eye_closed_threshold"] = median(right_closed_data)
                        print(f"[cal] Right blink range: {right_closed_data[0]}-{right_closed_data[-1]}, median: {median(right_closed_data)}")
                    print(f"[cal] center_offset={data.get('center_offset')}, "
                          f"left_threshold={data.get('left_threshold')}, "
                          f"right_threshold={data.get('right_threshold')}")
                    break
            else:
                # No face detected - show message
                cv2.putText(frame, "No face detected - stay in frame",
                            (frame_w // 2 - 250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Write frame to shared buffer
            encoded = _encode(frame, quality=65)
            with _buf_lock:
                _webcam_b64 = encoded
                _last_webcam = frame

        # ── Calibration complete ─────────────────────────────────
        cam.release()
        face_mesh.close()

        if _cal_stop.is_set():
            cal_progress.update({"active": False, "done": False})
            return

        # Save config
        with open(cfg_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[cal] Config saved to {cfg_path}")

        # Create and start GazeTracker
        tracker = GazeTracker(cfg_path)
        tracker.config["unity_camera"] = False
        tracker.config["scene_camera"] = False
        tracker.showCVWindow = False
        tracker.start()
        print("[cal] GazeTracker started")

        state["cal_step"] = 4
        state["cal_done"] = True
        state["running"] = True
        state["left_threshold"] = data.get("left_threshold", 0.35)
        state["right_threshold"] = data.get("right_threshold", 0.65)
        cal_progress.update({
            "active": False, "step_name": "done", "step_index": 6,
            "time_left": 0, "done": True, "error": "",
        })

    except Exception as e:
        print(f"[cal] Error: {traceback.format_exc()}")
        cal_progress.update({"active": False, "done": False, "error": str(e)})
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass


def _demo_calibration_worker():
    """Fake calibration for demo mode — just waits then starts DemoTracker."""
    global tracker
    total = CAL_GET_READY_TIME * 3 + CAL_DURATION_EACH_STEP * 5 + 1
    steps = [
        ("get_ready", 0, CAL_GET_READY_TIME),
        ("center", 1, CAL_DURATION_EACH_STEP),
        ("left", 2, CAL_DURATION_EACH_STEP),
        ("right", 3, CAL_DURATION_EACH_STEP),
        ("prep_left_blink", 3, CAL_GET_READY_TIME),
        ("left_blink", 4, CAL_DURATION_EACH_STEP),
        ("prep_right_blink", 4, CAL_GET_READY_TIME),
        ("right_blink", 5, CAL_DURATION_EACH_STEP),
    ]
    cal_progress.update({
        "active": True, "done": False, "error": "",
        "total_duration": total, "total_elapsed": 0,
    })

    elapsed = 0.0
    for step_name, step_idx, duration in steps:
        if _cal_stop.is_set():
            cal_progress.update({"active": False, "done": False})
            return
        cal_progress["step_name"] = step_name
        cal_progress["step_index"] = step_idx
        step_start = time.time()
        while time.time() - step_start < duration:
            if _cal_stop.is_set():
                cal_progress.update({"active": False, "done": False})
                return
            now_elapsed = elapsed + (time.time() - step_start)
            cal_progress["total_elapsed"] = now_elapsed
            cal_progress["time_left"] = duration - (time.time() - step_start)
            time.sleep(0.1)
        elapsed += duration

    tracker = DemoTracker()
    tracker.start()
    state["cal_step"] = 4
    state["cal_done"] = True
    state["running"] = True
    cal_progress.update({
        "active": False, "step_name": "done", "step_index": 6,
        "time_left": 0, "done": True, "error": "",
    })


def _encode(frame, quality=60):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def _pip_overlay(base, pip, scale=0.26, margin=8):
    base = base.copy()
    bh, bw = base.shape[:2]
    pw = int(bw * scale)
    ph = int(pip.shape[0] * pw / pip.shape[1])
    pip_r = cv2.resize(pip, (pw, ph))
    x1, y1 = bw - pw - margin, margin
    cv2.rectangle(base, (x1 - 2, y1 - 2), (x1 + pw + 2, y1 + ph + 2), (20, 20, 20), 2)
    base[y1:y1 + ph, x1:x1 + pw] = pip_r
    return base


# ── Background workers ───────────────────────────────────────────────────────
def _webcam_worker():
    global _webcam_b64, _last_webcam
    while True:
        try:
            t = tracker
            if t is not None:
                wf = None
                pipeline_running = getattr(t, "running", False)
                if pipeline_running:
                    # Preferred source: tracker.current_control_frame — always
                    # holds the LATEST annotated frame (MediaPipe landmarks, eye
                    # boxes, pose vector). Updated on every tracker iteration.
                    # This gives the dashboard the tracked view users expect.
                    cf = getattr(t, "current_control_frame", None)
                    if cf is not None and getattr(cf, "size", 0) > 0:
                        wf = cf
                    else:
                        # Secondary: try the one-shot queue (rarely populated).
                        try:
                            wf = t.get_frame()
                        except Exception:
                            pass
                        # Last resort: raw camera frame (no annotations).
                        # Used only if tracker hasn't produced a frame yet.
                        if wf is None:
                            cam = getattr(t, "cam", None)
                            if cam is not None:
                                try:
                                    ret, raw = cam.read()
                                    if ret and raw is not None:
                                        wf = cv2.flip(raw, 1)
                                except Exception:
                                    pass
                else:
                    cam = getattr(t, "cam", None)
                    if cam is not None:
                        try:
                            ret, raw = cam.read()
                            if ret and raw is not None:
                                wf = cv2.flip(raw, 1)
                        except Exception:
                            pass
                if wf is not None:
                    encoded = _encode(wf, quality=65)
                    with _buf_lock:
                        _last_webcam = wf
                        _webcam_b64 = encoded
        except Exception as e:
            print(f"[webcam worker] {e}")
        time.sleep(0.04)


def _unity_worker():
    """
    Legacy worker — kept as a no-op mirror of _webcam_b64.

    Before Phase 4, this worker could receive Unity-rendered frames over a
    TCP socket and compose a PiP overlay. The WebGL bridge replaced that
    path entirely: Unity now renders in the browser, not in a separate
    process. We keep the worker alive to preserve _unity_b64 as a fallback
    reference used by /ws/camera; it simply mirrors the webcam buffer.
    """
    global _unity_b64, unity_conn
    while True:
        try:
            with _buf_lock:
                _unity_b64 = _webcam_b64
            unity_conn = False
        except Exception as e:
            print(f"[unity worker] {e}")
        time.sleep(0.05)


threading.Thread(target=_webcam_worker, daemon=True).start()
threading.Thread(target=_unity_worker, daemon=True).start()


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Cerebruh Gaze Dashboard")

# ── Unity WebGL build serving ────────────────────────────────────────────────
_UNITY_BUILD = _HERE / "static" / "unity" / "Build"

# MIME type map for Unity's Brotli-compressed build files (suffixes include .br).
# Used when the build was exported with Compression Format = Brotli.
_BR_MIME = {
    ".wasm.br":         "application/wasm",
    ".js.br":           "application/javascript",
    ".data.br":         "application/octet-stream",
    ".symbols.json.br": "application/octet-stream",
}

# MIME type map for uncompressed Unity WebGL build files.
# Used when the build was exported without compression (batch mode default).
_PLAIN_MIME = {
    ".wasm":         "application/wasm",
    ".framework.js": "application/javascript",
    ".loader.js":    "application/javascript",
    ".data":         "application/octet-stream",
    ".js":           "application/javascript",
}


@app.get("/static/unity/Build/{path:path}")
async def serve_unity_build(path: str):
    """Serve Unity WebGL build files with correct MIME types.

    Handles both Brotli-compressed builds (.wasm.br, .js.br, .data.br) and
    uncompressed builds (.wasm, .framework.js, .data) produced by batch-mode
    exports where Brotli compression is not applied.
    """
    file_path = _UNITY_BUILD / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Build file not found: {path}")

    # Brotli-compressed files: longer compound suffixes checked first
    if path.endswith(".br"):
        media_type = "application/octet-stream"
        for suffix, mime in _BR_MIME.items():
            if path.endswith(suffix):
                media_type = mime
                break
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            headers={"Content-Encoding": "br"},
        )

    # Uncompressed files: check compound suffixes (e.g. .framework.js) before .js
    media_type = "application/octet-stream"
    for suffix, mime in _PLAIN_MIME.items():
        if path.endswith(suffix):
            media_type = mime
            break
    return FileResponse(path=str(file_path), media_type=media_type)


app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_HERE / "static" / "index.html"))


@app.get("/api/state")
async def get_state():
    return JSONResponse({
        "demo_mode": DEMO_MODE,
        "unity_available": UNITY_AVAILABLE,
        **state,
    })


@app.post("/api/calibrate/start")
async def calibrate_start():
    global _cal_thread
    gaze_history.clear()
    try:
        # Stop any existing calibration
        _cal_stop.set()
        if _cal_thread and _cal_thread.is_alive():
            _cal_thread.join(timeout=3)
        _cal_stop.clear()

        state["cal_step"] = 1
        if DEMO_MODE:
            _cal_thread = threading.Thread(target=_demo_calibration_worker, daemon=True)
        else:
            _cal_thread = threading.Thread(target=_calibration_worker, daemon=True)
        _cal_thread.start()
        return JSONResponse({"ok": True, "step": 1})
    except Exception:
        err = traceback.format_exc()
        print(err)
        return JSONResponse({"ok": False, "error": err[:200]}, status_code=500)


@app.post("/api/calibrate/reset")
async def calibrate_reset():
    global _cal_thread
    _cal_stop.set()
    if _cal_thread and _cal_thread.is_alive():
        _cal_thread.join(timeout=3)
    _cal_stop.clear()
    state["cal_step"] = 0
    cal_progress.update({"active": False, "done": False, "error": ""})
    return JSONResponse({"ok": True, "step": 0})


@app.post("/api/stop")
async def stop_tracking():
    global tracker, _cal_thread
    # Stop calibration if active
    _cal_stop.set()
    if _cal_thread and _cal_thread.is_alive():
        _cal_thread.join(timeout=3)
    _cal_stop.clear()
    cal_progress.update({"active": False, "done": False, "error": ""})

    if tracker:
        try:
            tracker.stop()
        except Exception:
            pass
        try:
            tracker.deinit()
        except Exception:
            pass
        tracker = None
    gaze_history.clear()
    state.update({"cal_step": 0, "cal_done": False, "running": False})
    return JSONResponse({"ok": True})


@app.post("/api/recenter")
async def recenter():
    if state["running"] and tracker and not DEMO_MODE:
        if hasattr(tracker, "value_buffer"):
            tracker.config["center_offset"] = \
                -tracker.value_buffer["gaze_horizontal_full"][-1]
            tracker.config["head_rotation_center_offset"] = \
                -tracker.value_buffer["head_rotation"][-1]
            print("[recenter] Applied.")
    return JSONResponse({"ok": True})


@app.post("/api/config")
async def update_config(request: Request):
    global _show_unity_feed
    body = await request.json()
    print(f"[config] Received: {body}")

    if "controller_mode" in body:
        state["controller_mode"] = body["controller_mode"]
    if "cruise_control" in body:
        state["cruise_control"] = body["cruise_control"]
    if "left_threshold" in body:
        state["left_threshold"] = body["left_threshold"]
    if "right_threshold" in body:
        state["right_threshold"] = body["right_threshold"]
    if "unity_feed" in body:
        state["unity_feed"] = body["unity_feed"]
        _show_unity_feed = body["unity_feed"]
        print(f"[config] Unity feed = {_show_unity_feed}")

    # Apply to tracker
    if state["running"] and tracker and not DEMO_MODE:
        tracker.config["controller_mode"] = state["controller_mode"]
        tracker.config["cruise_control_enabled"] = state["cruise_control"]
        tracker.config["left_threshold"] = state["left_threshold"]
        tracker.config["right_threshold"] = state["right_threshold"]
    return JSONResponse({"ok": True})


# ── WebSocket: stream gaze data ─────────────────────────────────────────────
@app.websocket("/ws/gaze")
async def ws_gaze(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = {}
            if state["running"] and tracker:
                raw = tracker.get_gaze_async()
                if raw:
                    gaze_history.append(raw)
                    data = raw

            history = list(gaze_history)
            payload = {
                "gaze": data,
                "history": history[-MAX_HISTORY:],
                "state": {**state, "unity_conn": unity_conn, "demo_mode": DEMO_MODE},
                "cal_progress": dict(cal_progress),
            }
            await ws.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/gaze] {e}")


# ── WebSocket: stream camera frames ─────────────────────────────────────────
@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with _buf_lock:
                if state.get("unity_feed") and state.get("cal_done"):
                    b64 = _unity_b64
                else:
                    b64 = _webcam_b64
            if b64:
                await ws.send_text(b64)
            await asyncio.sleep(0.06)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/camera] {e}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print(f"  Cerebruh Gaze Dashboard  |  DEMO={DEMO_MODE}")
    print("  http://127.0.0.1:8050")
    print("=" * 56)
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
