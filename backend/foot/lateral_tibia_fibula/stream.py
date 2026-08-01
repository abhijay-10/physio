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

st.subheader("📸 Axoris Lateral Tibia/Fibula Positioner")
st.info("🎥 Status: Auto-detecting the leg resting horizontally in lateral view.")

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
        if len(self.hip_q) > 0:
            last_hip = self.hip_q[-1]
            last_knee = self.knee_q[-1]
            last_ankle = self.ankle_q[-1]
            # Ignore jitter smaller than 10 pixels for absolute stability
            if math.hypot(hip[0] - last_hip[0], hip[1] - last_hip[1]) < 10: hip = last_hip
            if math.hypot(knee[0] - last_knee[0], knee[1] - last_knee[1]) < 10: knee = last_knee
            if math.hypot(ankle[0] - last_ankle[0], ankle[1] - last_ankle[1]) < 10: ankle = last_ankle
            
        self.hip_q.append(hip)
        self.knee_q.append(knee)
        self.ankle_q.append(ankle)
        
        # Give higher weight to more recent frames for responsive but stable smoothing
        weights = np.linspace(0.5, 1.0, len(self.hip_q))
        s_hip = tuple(np.average(self.hip_q, axis=0, weights=weights).astype(int))
        s_knee = tuple(np.average(self.knee_q, axis=0, weights=weights).astype(int))
        s_ankle = tuple(np.average(self.ankle_q, axis=0, weights=weights).astype(int))
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

if "active_lateral_tibia_camera" not in st.session_state:
    st.session_state.active_lateral_tibia_camera = None

if "current_lateral_tibia_cam_idx" not in st.session_state:
    st.session_state.current_lateral_tibia_cam_idx = -1

if st.session_state.current_lateral_tibia_cam_idx != cam_choice:
    if st.session_state.active_lateral_tibia_camera is not None:
        st.session_state.active_lateral_tibia_camera.stop()
    st.session_state.active_lateral_tibia_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_lateral_tibia_cam_idx = cam_choice

if "last_landmarks" not in st.session_state:
    st.session_state.last_landmarks = None
if "pose_lost_counter" not in st.session_state:
    st.session_state.pose_lost_counter = 0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
engine_right = AdaptiveLegStabilizer(window_size=10, box_alpha=0.1)
engine_left = AdaptiveLegStabilizer(window_size=10, box_alpha=0.1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.1, 
    min_tracking_confidence=0.1,
    min_pose_presence_confidence=0.1
)
detector = vision.PoseLandmarker.create_from_options(options)
vs = st.session_state.active_lateral_tibia_camera

def check_leg_state(landmarks, h, w, hip_idx, knee_idx, ankle_idx, foot_idx, engine):
    # Removed visibility check because MediaPipe's confidence in partial body landmarks fluctuates wildly,
    # and the geometry span check is sufficient to prevent hallucinations.
        
    raw_hip = (int((1 - landmarks[hip_idx].x) * w), int(landmarks[hip_idx].y * h))
    raw_knee = (int((1 - landmarks[knee_idx].x) * w), int(landmarks[knee_idx].y * h))
    raw_ankle = (int((1 - landmarks[ankle_idx].x) * w), int(landmarks[ankle_idx].y * h))
    raw_foot = (int((1 - landmarks[foot_idx].x) * w), int(landmarks[foot_idx].y * h))
    
    pixel_span = math.hypot(raw_ankle[0] - raw_hip[0], raw_ankle[1] - raw_hip[1])
    # Ignore any detected "leg" that is smaller than 250 pixels (e.g. blanket wrinkles)
    if pixel_span < 250:
        return None

    hip_pt, knee_pt, ankle_pt = engine.smooth(raw_hip, raw_knee, raw_ankle)
    
    # 1. Straight leg check (knee angle)
    angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
    angle_ok = 175.0 <= angle <= 180.0
    
    # 2. Horizontal orientation check (leg should be resting horizontally)
    dy = abs(ankle_pt[1] - hip_pt[1])
    dx = abs(ankle_pt[0] - hip_pt[0])
    leaning_angle = np.degrees(np.arctan2(dy, dx + 1e-6))
    leaning_ok = leaning_angle <= 25.0 # Max 25 degrees tilt from perfectly horizontal
    
    # 3. Foot pointing horizontally (not pointing up)
    foot_angle = calculate_angle_3pt(knee_pt, ankle_pt, raw_foot)
    foot_ok = foot_angle > 130.0 # If pointing up (dorsiflexed), angle drops. Neutral is >130.
    
    motion_ok = True
    prev_key = f"prev_knee_{hip_idx}"
    if prev_key in st.session_state and st.session_state[prev_key] is not None:
        dist = np.linalg.norm(np.array(knee_pt) - np.array(st.session_state[prev_key]))
        if dist > 80.0: # Increased from 40.0 to prevent flickering
            motion_ok = False
    st.session_state[prev_key] = knee_pt
    
    xs = [hip_pt[0], knee_pt[0], ankle_pt[0]]
    ys = [hip_pt[1], knee_pt[1], ankle_pt[1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    span = max(max_x - min_x, max_y - min_y)
    raw_box_size = int(np.clip(span + 250, 400, 900))
    raw_tx = center_x - (raw_box_size // 2)
    raw_ty = center_y - (raw_box_size // 2)
    
    tx, ty, dbs = engine.smooth_box(raw_tx, raw_ty, raw_box_size)
    tx = np.clip(tx, 10, w - dbs - 10)
    ty = np.clip(ty, 10, h - dbs - 10)
    
    avg_z = (landmarks[hip_idx].z + landmarks[knee_idx].z + landmarks[ankle_idx].z) / 3.0
    
    return {
        "visible": True,
        "span": pixel_span,
        "avg_z": avg_z,
        "angle": angle,
        "leaning_angle": leaning_angle,
        "angle_ok": angle_ok,
        "leaning_ok": leaning_ok,
        "foot_ok": foot_ok,
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
    cv2.line(frame, hip_pt, knee_pt, (255, 100, 100), 2, cv2.LINE_AA)
    cv2.line(frame, knee_pt, ankle_pt, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, hip_pt, 8, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, knee_pt, 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, ankle_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Knee: {int(state['angle'])} deg", (knee_pt[0] + 20, knee_pt[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tilt: {int(state['leaning_angle'])} deg", (knee_pt[0] + 20, knee_pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

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
                if st.session_state.last_landmarks is not None and st.session_state.pose_lost_counter < 300:
                    landmarks = st.session_state.last_landmarks
                    st.session_state.pose_lost_counter += 1
                else:
                    st.session_state.last_landmarks = None

            color = (0, 0, 255)
            status_text = "ALIGN LEG HORIZONTALLY"
            feedback_details = []

            if landmarks is not None:
                right_state = check_leg_state(landmarks, h, w, 23, 25, 27, 31, engine_right)
                left_state = check_leg_state(landmarks, h, w, 24, 26, 28, 32, engine_left)

                # Find the primary leg that is visible (prioritize the one that is closest to correct, or just pick the best)
                # If both are visible, we can evaluate whichever is better, or just evaluate if ANY leg is correct.
                primary_state = None
                if right_state and left_state:
                    # Pick the leg that is physically closest to the camera (largest pixel span due to perspective)
                    if right_state["span"] > left_state["span"]:
                        primary_state = right_state
                    else:
                        primary_state = left_state
                elif right_state:
                    primary_state = right_state
                elif left_state:
                    primary_state = left_state

                if primary_state:
                    final_ok = primary_state["angle_ok"] and primary_state["leaning_ok"] and primary_state["foot_ok"] and primary_state["motion_ok"]
                    
                    if final_ok:
                        status_text = "LATERAL LEG: OK"
                        color = (0, 255, 0)
                        status_msg.success(f"✅ Pose is correct. While doing X-ray, take camera angle from upper view.")
                    else:
                        status_text = "LATERAL LEG: ADJUST POSTURE"
                        color = (0, 0, 255)
                        if not primary_state["angle_ok"]:
                            feedback_details.append("Straighten the knee fully.")
                        if not primary_state["leaning_ok"]:
                            feedback_details.append("Leg must be resting horizontally (tilt < 25 deg).")
                        if not primary_state["foot_ok"]:
                            feedback_details.append("Relax foot down, don't point toes up.")
                        if not primary_state["motion_ok"]:
                            feedback_details.append("Hold completely still.")
                        
                        status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")
                else:
                    status_text = "ADJUST CAMERA"
                    color = (0, 165, 255)
                    status_msg.warning("⚠️ No leg clearly visible. Align in camera.")

                if 'global_telemetry' in globals():
                    if not primary_state:
                        global_telemetry['status'] = "bad"
                        global_telemetry['message'] = "Align leg clearly in camera view."
                        global_telemetry['accuracy'] = 45
                    elif final_ok:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Pose is correct. While doing X-ray, take camera angle from upper view."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        global_telemetry['message'] = f"Warning: {' | '.join(feedback_details)}"

                if primary_state:
                    draw_leg(display_frame, primary_state, color)

                cv2.putText(display_frame, "TARGET: LATERAL LEG", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "bad"
                    global_telemetry['message'] = "Warning: Pose lost. Align your leg inside the camera window bounds..."
                    global_telemetry['accuracy'] = 0
                status_msg.warning("⚠️ Pose lost. Align your leg inside the camera window bounds...")
                cv2.putText(display_frame, "ADJUST CAMERA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass
