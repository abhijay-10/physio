import os
#### 180 degree knee angle straight leg
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import streamlit as st
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STREAMLIT INLINE WORKSPACE INITIALIZATION ---
st.subheader("📸 Boundary-Locked PA Patella / Knee Positioner")

cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

frame_window = st.empty()  
status_msg = st.empty()

# ==========================================
# 1. LIVE HARDWARE STREAM (Thread-Isolated Buffer)
# ==========================================
class LiveVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

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
                if grabbed: self.frame = frame
            time.sleep(0.01) 

    def read(self):
        with self.read_lock:
            if self.grabbed and self.frame is not None: 
                return self.frame.copy()
            return None

    def stop(self):
        self.started = False
        if self.stream.isOpened(): self.stream.release()

# ==========================================
# 2. PERSISTENT HISTORICAL BUFFER FILTER
# ==========================================
class FallbackBufferStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.hip_history = deque(maxlen=window_size)
        self.knee_history = deque(maxlen=window_size)
        self.ankle_history = deque(maxlen=window_size)
        self.box_history = deque(maxlen=window_size)

    def process(self, hip, knee, ankle, hip_vis, knee_vis, ankle_vis, frame_height):
        """Filters out ghost tracking anomalies that appear way too high up in the frame"""
        # CRITICAL SAFETY REJECTION CHECK: If tracking locks on something in the upper 35% 
        # of the frame while you are lying flat, flag it as a false background positive.
        is_anomaly = knee[1] < int(frame_height * 0.35)

        if hip_vis > 0.40 and not is_anomaly: self.hip_history.append(hip)
        if knee_vis > 0.40 and not is_anomaly: self.knee_history.append(knee)
        if ankle_vis > 0.40 and not is_anomaly: self.ankle_history.append(ankle)

        final_hip = tuple(np.mean(self.hip_history, axis=0).astype(int)) if self.hip_history else hip
        final_knee = tuple(np.mean(self.knee_history, axis=0).astype(int)) if self.knee_history else knee
        final_ankle = tuple(np.mean(self.ankle_history, axis=0).astype(int)) if self.ankle_history else ankle

        return final_hip, final_knee, final_ankle

    def smooth_box(self, current_corner):
        self.box_history.append(current_corner)
        arr = np.array(self.box_history)
        return int(np.mean(arr[:, 0])), int(np.mean(arr[:, 1]))

def calculate_angle_3pt(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Session state checks
if "active_knee_camera" not in st.session_state:
    st.session_state.active_knee_camera = None
if "current_knee_cam_idx" not in st.session_state:
    st.session_state.current_knee_cam_idx = -1

if st.session_state.current_knee_cam_idx != cam_choice:
    if st.session_state.active_knee_camera is not None:
        st.session_state.active_knee_camera.stop()
    st.session_state.active_knee_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_knee_cam_idx = cam_choice

# ==========================================
# 3. CONFIGURE DETECTOR CONTEXT
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
fallback_engine = FallbackBufferStabilizer(window_size=10)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.35, min_tracking_confidence=0.55        
)
detector = vision.PoseLandmarker.create_from_options(options)

vs = st.session_state.active_knee_camera

# ==========================================
# 4. MAIN PROCESSING PIPELINE
# ==========================================
try:
    if vs.frame is None:
        st.error(f"❌ Camera Source Index {cam_choice} offline.")
    else:
        while vs.started:
            frame = vs.read()
            if frame is None: continue

            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape

            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            current_timestamp_ms = int(time.time() * 1000)
            if \'last_timestamp_ms\' not in locals(): last_timestamp_ms = 0
            if current_timestamp_ms <= last_timestamp_ms: current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            result = detector.detect_for_video(mp_image, current_timestamp_ms)

            status_text = "TILT CAMERA DOWN - AIM AT MATTRESS"
            color = (0, 0, 255)
            side_label = "SCANNING LOWER VIEWPORT ZONE..."
            feedback_details = []

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                left_score = landmarks[23].visibility + landmarks[25].visibility + landmarks[27].visibility
                right_score = landmarks[24].visibility + landmarks[26].visibility + landmarks[28].visibility

                if left_score > right_score:
                    hip_lm, knee_lm, ankle_lm = landmarks[23], landmarks[25], landmarks[27]
                    heel_lm, toe_lm = landmarks[29], landmarks[31]
                    side_label = "PA RADIOLOGY: LEFT KNEE PROFILE"
                else:
                    hip_lm, knee_lm, ankle_lm = landmarks[24], landmarks[26], landmarks[28]
                    heel_lm, toe_lm = landmarks[30], landmarks[32]
                    side_label = "PA RADIOLOGY: RIGHT KNEE PROFILE"

                raw_hip = (int((1 - hip_lm.x) * w), int(hip_lm.y * h))
                raw_knee = (int((1 - knee_lm.x) * w), int(knee_lm.y * h))
                raw_ankle = (int((1 - ankle_lm.x) * w), int(ankle_lm.y * h))

                # Pass through the fallback engine with height variable context
                hip_pt, knee_pt, ankle_pt = fallback_engine.process(
                    raw_hip, raw_knee, raw_ankle, 
                    hip_lm.visibility, knee_lm.visibility, ankle_lm.visibility, h
                )

                # Dynamically calculate box sizing footprint
                limb_pixel_distance = np.linalg.norm(np.array(hip_pt) - np.array(ankle_pt))
                dynamic_box_size = int(np.clip(limb_pixel_distance * 0.55, 190, 320))

                t_x = knee_pt[0] - (dynamic_box_size // 2)
                t_y = knee_pt[1] - (dynamic_box_size // 2)
                target_x, target_y = fallback_engine.smooth_box((t_x, t_y))

                target_x = np.clip(target_x, 10, w - dynamic_box_size - 10)
                target_y = np.clip(target_y, 10, h - dynamic_box_size - 10)

                # Posture checks
                extension_angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
                extension_ok = 160.0 <= extension_angle <= 185.0
                
                # Enforces back-pose layout criteria context (Prone rule)
                is_prone_back_pose = (heel_lm.y < ankle_lm.y) and (toe_lm.y > heel_lm.y)
                
                # Check for 5-10 degree lateral rotation out of centerline axis
                heel_rotated_ok = abs(heel_lm.x - ankle_lm.x) > 0.007

                # Safety Check: Reject if knee joint center is too high up (indicates ghost mapping)
                is_valid_lower_zone = knee_pt[1] > int(h * 0.35)

                good_posture = extension_ok and is_prone_back_pose and heel_rotated_ok and is_valid_lower_zone
                
                if good_posture:
                    status_text = "RIGHT POSTURE"
                    color = (0, 255, 0)
                    status_msg.success(f"✅ LOCK STABLE: PA Knee Popliteal Space Aligned ({extension_angle:.1f}°)")
                else:
                    status_text = "WRONG POSTURE"
                    color = (0, 0, 255)
                    if not is_valid_lower_zone or not is_prone_back_pose:
                        feedback_details.append("Lie flat on your STOMACH (Face down, back of knee up)")
                    if not extension_ok and is_valid_lower_zone: 
                        feedback_details.append(f"Extend leg completely flat ({extension_angle:.1f}° found)")
                    if not heel_rotated_ok and is_valid_lower_zone: 
                        feedback_details.append("Turn your heel 5-10° outwards away from centerline")
                    status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")

                # --- TELEMETRY updates for voice assistance ---
                if 'global_telemetry' in globals():
                    if good_posture:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect alignment. Keep holding."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        if not is_valid_lower_zone or not is_prone_back_pose:
                            global_telemetry['message'] = "Warning: Lie flat on your STOMACH (Face down, back of knee up)"
                        elif not extension_ok:
                            global_telemetry['message'] = f"Warning: Extend leg completely flat"
                        elif not heel_rotated_ok:
                            global_telemetry['message'] = "Warning: Turn your heel 5 to 10 degrees outwards"

                # Render tracking assets
                cv2.rectangle(display_frame, (target_x, target_y), (target_x + dynamic_box_size, target_y + dynamic_box_size), color, 2, cv2.LINE_AA)
                cv2.line(display_frame, (knee_pt[0] - 40, knee_pt[1]), (knee_pt[0] + 40, knee_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(display_frame, (knee_pt[0], knee_pt[1] - 40), (knee_pt[0], knee_pt[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)

                # Connect skeletal bones vectors lines path
                cv2.line(display_frame, hip_pt, knee_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, knee_pt, ankle_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(display_frame, hip_pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, knee_pt, 7, color, -1, cv2.LINE_AA)   
                cv2.circle(display_frame, ankle_pt, 5, (0, 255, 255), -1, cv2.LINE_AA)

                cv2.putText(display_frame, side_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                status_msg.warning("⚠️ Adjust camera framing. Center your lower body leg profile flat inside layout window frame...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # --- TELEMETRY updates for voice assistance ---
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "calibrating"
                    global_telemetry['message'] = "Warning: Center your leg flat inside camera view"
                    global_telemetry['accuracy'] = 10

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass