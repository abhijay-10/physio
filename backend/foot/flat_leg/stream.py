import os
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import streamlit as st
import math
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_angle_3pt(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

st.subheader("📸 Axoris Flat Leg AP Positioner")
st.info("🎥 Status: Auto-detecting BOTH legs lying flat on the surface.")

frame_window = st.empty()  
status_msg = st.empty()

class LiveVideoStream:
    def __init__(self, src=2): 
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.new_frame = False

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed: 
                    self.frame = frame
                    self.new_frame = True
            time.sleep(0.002) 

    def has_new_frame(self):
        with self.read_lock:
            return self.new_frame

    def read(self):
        with self.read_lock:
            if self.grabbed and self.frame is not None: 
                self.new_frame = False
                return self.frame.copy()
            return None

    def stop(self):
        self.started = False
        if self.stream.isOpened(): self.stream.release()

class AdaptiveLegStabilizer:
    def __init__(self, window_size=3, box_alpha=0.4):
        self.window_size = window_size
        self.box_alpha = box_alpha
        self.hip_q = deque(maxlen=window_size)
        self.knee_q = deque(maxlen=window_size)
        self.ankle_q = deque(maxlen=window_size)
        self.prev_box_coords = None

    def smooth(self, hip, knee, ankle):
        self.hip_q.append(hip)
        self.knee_q.append(knee)
        self.ankle_q.append(ankle)
        s_hip = tuple(np.mean(self.hip_q, axis=0).astype(int))
        s_knee = tuple(np.mean(self.knee_q, axis=0).astype(int))
        s_ankle = tuple(np.mean(self.ankle_q, axis=0).astype(int))
        return s_hip, s_knee, s_ankle

    def smooth_box(self, tx, ty, size):
        if self.prev_box_coords is None:
            self.prev_box_coords = np.array([tx, ty, size], dtype=float)
        else:
            curr = np.array([tx, ty, size], dtype=float)
            self.prev_box_coords = self.prev_box_coords * (1 - self.box_alpha) + curr * self.box_alpha
        out = self.prev_box_coords.astype(int)
        return out[0], out[1], out[2]

cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

if "active_foot_camera" not in st.session_state:
    st.session_state.active_foot_camera = None
if "current_foot_cam_idx" not in st.session_state:
    st.session_state.current_foot_cam_idx = -1

if st.session_state.current_foot_cam_idx != cam_choice:
    if st.session_state.active_foot_camera is not None:
        st.session_state.active_foot_camera.stop()
    st.session_state.active_foot_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_foot_cam_idx = cam_choice

if "last_landmarks" not in st.session_state:
    st.session_state.last_landmarks = None
if "pose_lost_counter" not in st.session_state:
    st.session_state.pose_lost_counter = 0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
engine_right = AdaptiveLegStabilizer(window_size=3, box_alpha=0.4)
engine_left = AdaptiveLegStabilizer(window_size=3, box_alpha=0.4)

# WE DROP CONFIDENCE TO 0.1 TO ALLOW PARTIAL BODY DETECTION (LEGS WITHOUT HEAD)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.1, 
    min_tracking_confidence=0.1,
    min_pose_presence_confidence=0.1
)
detector = vision.PoseLandmarker.create_from_options(options)
vs = st.session_state.active_foot_camera

def check_leg_state(landmarks, h, w, hip_idx, knee_idx, ankle_idx, engine):
    # Base visibility check
    if (landmarks[hip_idx].visibility < 0.05 or 
        landmarks[knee_idx].visibility < 0.05 or 
        landmarks[ankle_idx].visibility < 0.05):
        return None
        
    raw_hip = (int((1 - landmarks[hip_idx].x) * w), int(landmarks[hip_idx].y * h))
    raw_knee = (int((1 - landmarks[knee_idx].x) * w), int(landmarks[knee_idx].y * h))
    raw_ankle = (int((1 - landmarks[ankle_idx].x) * w), int(landmarks[ankle_idx].y * h))
    
    # ANTI-HALLUCINATION FILTER: The leg must physically span at least 150 pixels vertically or diagonally
    # If the AI hallucinates a tiny leg on a shirt wrinkle, the distance will be very small.
    pixel_span = math.hypot(raw_ankle[0] - raw_hip[0], raw_ankle[1] - raw_hip[1])
    if pixel_span < 150:
        return None

    hip_pt, knee_pt, ankle_pt = engine.smooth(raw_hip, raw_knee, raw_ankle)
    
    angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
    angle_ok = 170.0 <= angle <= 180.0 # Relaxed to 170 to catch minor knee lifts but avoid flickering
    
    leaning_angle = np.degrees(np.arctan2(abs(ankle_pt[0] - hip_pt[0]), abs(ankle_pt[1] - hip_pt[1]) + 1e-6))
    leaning_ok = True # Allow lying horizontally or vertically
    
    motion_ok = True
    prev_key = f"prev_knee_{hip_idx}"
    if prev_key in st.session_state and st.session_state[prev_key] is not None:
        dist = np.linalg.norm(np.array(knee_pt) - np.array(st.session_state[prev_key]))
        # Increased threshold from 8.0 to 40.0 to prevent flickering from AI landmark jitter
        if dist > 40.0:
            motion_ok = False
    st.session_state[prev_key] = knee_pt
    
    xs = [hip_pt[0], knee_pt[0], ankle_pt[0]]
    ys = [hip_pt[1], knee_pt[1], ankle_pt[1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    span = max(max_x - min_x, max_y - min_y)
    raw_box_size = int(np.clip(span + 100, 300, 700))
    raw_tx = center_x - (raw_box_size // 2)
    raw_ty = center_y - (raw_box_size // 2)
    
    tx, ty, dbs = engine.smooth_box(raw_tx, raw_ty, raw_box_size)
    tx = np.clip(tx, 10, w - dbs - 10)
    ty = np.clip(ty, 10, h - dbs - 10)
    
    return {
        "visible": True,
        "angle": angle,
        "angle_ok": angle_ok,
        "leaning_ok": leaning_ok,
        "motion_ok": motion_ok,
        "hip_pt": hip_pt,
        "knee_pt": knee_pt,
        "ankle_pt": ankle_pt,
        "box": (tx, ty, dbs)
    }

def draw_leg(frame, state, color):
    if not state: return
    tx, ty, dbs = state["box"]
    hip_pt, knee_pt, ankle_pt = state["hip_pt"], state["knee_pt"], state["ankle_pt"]
    cv2.rectangle(frame, (tx, ty), (tx + dbs, ty + dbs), color, 2, cv2.LINE_AA)
    cv2.line(frame, (knee_pt[0] - 30, knee_pt[1]), (knee_pt[0] + 30, knee_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (knee_pt[0], knee_pt[1] - 30), (knee_pt[0], knee_pt[1] + 30), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, hip_pt, knee_pt, (255, 100, 100), 2, cv2.LINE_AA)
    cv2.line(frame, knee_pt, ankle_pt, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, hip_pt, 8, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, knee_pt, 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, ankle_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
    # Add text back near the knee so user can debug their angle
    cv2.putText(frame, f"Knee: {int(state['angle'])} deg", (knee_pt[0] + 20, knee_pt[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

last_timestamp_ms = 0
try:
    if vs.frame is None:
        st.error("❌ External Camera Index 2 offline. Verify hardware cable links.")
    else:
        while vs.started:
            if not vs.has_new_frame():
                time.sleep(0.002)
                continue

            frame = vs.read()
            if frame is None: continue
            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape

            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image_obj, timestamp_ms)
            landmarks = None

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                st.session_state.last_landmarks = landmarks
                st.session_state.pose_lost_counter = 0
            else:
                if st.session_state.last_landmarks is not None and st.session_state.pose_lost_counter < 30:
                    landmarks = st.session_state.last_landmarks
                    st.session_state.pose_lost_counter += 1
                else:
                    st.session_state.last_landmarks = None

            color = (0, 0, 255)
            status_text = "ALIGN BOTH LEGS FLAT"
            feedback_details = []

            if landmarks is not None:
                # Right leg on screen is mapped from left body landmarks (23, 25, 27)
                right_state = check_leg_state(landmarks, h, w, 23, 25, 27, engine_right)
                # Left leg on screen is mapped from right body landmarks (24, 26, 28)
                left_state = check_leg_state(landmarks, h, w, 24, 26, 28, engine_left)

                if right_state and left_state:
                    final_ok = (right_state["angle_ok"] and right_state["leaning_ok"] and right_state["motion_ok"] and
                                left_state["angle_ok"] and left_state["leaning_ok"] and left_state["motion_ok"])
                    
                    if final_ok:
                        status_text = "FLAT LEGS: OK"
                        color = (0, 255, 0)
                        status_msg.success(f"✅ FLAT LEGS LOCKED: Perfect Posture Detected.")
                    else:
                        status_text = "FLAT LEGS: ADJUST POSTURE"
                        color = (0, 0, 255)
                        if not right_state["angle_ok"] or not left_state["angle_ok"]:
                            feedback_details.append("Straighten both knees fully.")
                        if not right_state["leaning_ok"] or not left_state["leaning_ok"]:
                            feedback_details.append("Keep legs vertically aligned without tilting.")
                        if not right_state["motion_ok"] or not left_state["motion_ok"]:
                            feedback_details.append("Hold completely still.")
                        
                        feedback_details = list(dict.fromkeys(feedback_details))
                        status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")
                else:
                    status_text = "ADJUST CAMERA"
                    color = (0, 165, 255)
                    status_msg.warning("⚠️ Both legs must be fully visible and flat.")

                if 'global_telemetry' in globals():
                    if not (right_state and left_state):
                        global_telemetry['status'] = "bad"
                        global_telemetry['message'] = "Both legs must be fully visible and flat."
                        global_telemetry['accuracy'] = 45
                    elif final_ok:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect alignment. Keep holding."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        global_telemetry['message'] = f"Warning: {' | '.join(feedback_details)}"

                # Draw legs
                if right_state and left_state:
                    draw_color = color
                else:
                    draw_color = (255, 255, 255)
                
                draw_leg(display_frame, right_state, draw_color)
                draw_leg(display_frame, left_state, draw_color)

                cv2.putText(display_frame, "TARGET: BOTH LEGS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "bad"
                    global_telemetry['message'] = "Warning: Pose lost. Align your legs inside the camera window bounds..."
                    global_telemetry['accuracy'] = 0
                status_msg.warning("⚠️ Pose lost. Align your legs inside the camera window bounds...")
                cv2.putText(display_frame, "ADJUST CAMERA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass