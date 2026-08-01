import os
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
st.subheader("📸 AP Lumbar Spine Live Positioner (Knees Flexed Profile)")
st.info("🎥 Locked to External Webcam Hardware Target: Camera Index 2")

frame_window = st.empty()  
status_msg = st.empty()

# ==========================================
# 1. LIVE HARDWARE STREAM (Thread-Isolated Buffer)
# ==========================================
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
# 2. TEMPORAL HISTORY BUFFER FILTER (Anti-Flicker)
# ==========================================
class FallbackBufferStabilizer:
    def __init__(self, window_size=8):
        self.window_size = window_size
        self.shoulder_history = deque(maxlen=window_size)
        self.hip_history = deque(maxlen=window_size)
        self.knee_history = deque(maxlen=window_size)
        self.ankle_history = deque(maxlen=window_size)
        self.box_history = deque(maxlen=window_size)

    def process(self, shoulder, hip, knee, ankle, s_vis, h_vis, k_vis, a_vis):
        if s_vis > 0.35: self.shoulder_history.append(shoulder)
        if h_vis > 0.35: self.hip_history.append(hip)
        if k_vis > 0.35: self.knee_history.append(knee)
        if a_vis > 0.35: self.ankle_history.append(ankle)

        final_s = tuple(np.mean(self.shoulder_history, axis=0).astype(int)) if self.shoulder_history else shoulder
        final_h = tuple(np.mean(self.hip_history, axis=0).astype(int)) if self.hip_history else hip
        final_k = tuple(np.mean(self.knee_history, axis=0).astype(int)) if self.knee_history else knee
        final_a = tuple(np.mean(self.ankle_history, axis=0).astype(int)) if self.ankle_history else ankle

        return final_s, final_h, final_k, final_a

    def smooth_box(self, current_corner):
        self.box_history.append(current_corner)
        arr = np.array(self.box_history)
        return int(np.mean(arr[:, 0])), int(np.mean(arr[:, 1]))

def calculate_angle_3pt(a, b, c):
    """Calculates inner joint angle at vertex b"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Session state continuity checks
cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

if "active_lumbar_camera" not in st.session_state:
    st.session_state.active_lumbar_camera = None
if "current_lumbar_cam_idx" not in st.session_state:
    st.session_state.current_lumbar_cam_idx = -1

if st.session_state.current_lumbar_cam_idx != cam_choice:
    if st.session_state.active_lumbar_camera is not None:
        st.session_state.active_lumbar_camera.stop()
    st.session_state.active_lumbar_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_lumbar_cam_idx = cam_choice

# ==========================================
# 3. CONFIGURE RUNTIME PIPELINE
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker_full.task").replace("\\", "/")
fallback_engine = FallbackBufferStabilizer(window_size=8)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.25, min_tracking_confidence=0.50        
)
detector = vision.PoseLandmarker.create_from_options(options)

vs = st.session_state.active_lumbar_camera

# ==========================================
# 4. EXECUTION PROCESSING LOOP
try:
    if vs.frame is None:
        st.error("❌ External Camera Index 2 could not be found. Check your USB port connections.")
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

            status_text = "ALIGN PATIENT SUPINE WITH KNEES BENT"
            color = (0, 0, 255)
            side_label = "SCANNING LUMBAR SPINE REGION..."
            feedback_details = []

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Select the visible profile side facing the camera profile lens
                left_score = landmarks[11].visibility + landmarks[23].visibility + landmarks[25].visibility
                right_score = landmarks[12].visibility + landmarks[24].visibility + landmarks[26].visibility

                if left_score > right_score:
                    sh_lm, hip_lm, knee_lm, ankle_lm = landmarks[11], landmarks[23], landmarks[25], landmarks[27]
                    side_label = "AP LUMBAR REGION: LEFT LATERAL PROFILE"
                else:
                    sh_lm, hip_lm, knee_lm, ankle_lm = landmarks[12], landmarks[24], landmarks[26], landmarks[28]
                    side_label = "AP LUMBAR REGION: RIGHT LATERAL PROFILE"

                # Translate coordinates to display pixels
                raw_sh = (int((1 - sh_lm.x) * w), int(sh_lm.y * h))
                raw_hip = (int((1 - hip_lm.x) * w), int(hip_lm.y * h))
                raw_knee = (int((1 - knee_lm.x) * w), int(knee_lm.y * h))
                raw_ankle = (int((1 - ankle_lm.x) * w), int(ankle_lm.y * h))

                # Stabilize coordinate matrix vectors through the fallback filter engine
                sh_pt, hip_pt, knee_pt, ankle_pt = fallback_engine.process(
                    raw_sh, raw_hip, raw_knee, raw_ankle,
                    sh_lm.visibility, hip_lm.visibility, knee_lm.visibility, ankle_lm.visibility
                )

                # Find the center of the Lumbar Spine (located between the chest line and pelvic hip bone)
                lumbar_x = int((sh_pt[0] + hip_pt[0] * 1.5) / 2.5)
                lumbar_y = int((sh_pt[1] + hip_pt[1] * 1.2) / 2.2)
                lumbar_center = (lumbar_x, lumbar_y)

                # Dynamically size the lumbar box field based on torso torso pixel scale dimensions
                torso_scale = np.linalg.norm(np.array(sh_pt) - np.array(hip_pt))
                dynamic_box_w = int(np.clip(torso_scale * 0.70, 220, 360))
                dynamic_box_h = int(np.clip(torso_scale * 0.50, 160, 260))

                # Smooth box corners translation path
                t_x = lumbar_center[0] - (dynamic_box_w // 2)
                t_y = lumbar_center[1] - (dynamic_box_h // 2)
                target_x, target_y = fallback_engine.smooth_box((t_x, t_y))

                # Screen edge safety overflow clippings
                target_x = np.clip(target_x, 10, w - dynamic_box_w - 10)
                target_y = np.clip(target_y, 10, h - dynamic_box_h - 10)

                # ==========================================
                # CLINICAL AP LUMBAR POSTURE VERIFICATIONS
                # ==========================================
                # Check 1: Knee joint flexion check to flatten lordosis curve (Target: 115° to 145°)
                knee_flexion_angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
                knees_flexed_ok = 112.0 <= knee_flexion_angle <= 148.0

                # Check 2: Supine Torso Baseline horizontal check (Torso must stay level on the table)
                is_supine = abs(sh_pt[1] - hip_pt[1]) < int(h * 0.18) and hip_pt[1] > int(h * 0.40)

                good_posture = knees_flexed_ok and is_supine
                
                if good_posture:
                    status_text = "RIGHT POSTURE"
                    color = (0, 255, 0) # Green position locked
                    status_msg.success(f"✅ POSITION PERFECT: Lumbar Curve Flattened. Knee Flexion at {knee_flexion_angle:.1f}°")
                else:
                    status_text = "WRONG POSTURE"
                    color = (0, 0, 255) # Red target indicator
                    if not is_supine:
                        feedback_details.append("Lie flat on your BACK (Supine position level on the mattress)")
                    if not knees_flexed_ok and is_supine:
                        feedback_details.append(f"Bend your knees closer to a 120° position ({knee_flexion_angle:.1f}° found)")
                    status_msg.error(f"❌ Fix: {' | '.join(feedback_details)}")

                # Draw rectangular collimation window centered over Lumbar vertebra spaces
                cv2.rectangle(display_frame, (target_x, target_y), (target_x + dynamic_box_w, target_y + dynamic_box_h), color, 2, cv2.LINE_AA)
                
                # Central Ray crosshair grid matching clinical specification sheets
                cv2.line(display_frame, (lumbar_center[0] - 40, lumbar_center[1]), (lumbar_center[0] + 40, lumbar_center[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(display_frame, (lumbar_center[0], lumbar_center[1] - 40), (lumbar_center[0], lumbar_center[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)

                # Render structural biomechanical connection links
                cv2.line(display_frame, sh_pt, hip_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, hip_pt, knee_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, knee_pt, ankle_pt, (255, 255, 255), 3, cv2.LINE_AA)
                
                cv2.circle(display_frame, sh_pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, lumbar_center, 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(display_frame, hip_pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, knee_pt, 7, color, -1, cv2.LINE_AA)
                cv2.circle(display_frame, ankle_pt, 5, (255, 0, 0), -1, cv2.LINE_AA)

                cv2.putText(display_frame, side_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
                
                if feedback_details:
                    y_offset = h - 40 * len(feedback_details) - 20
                    for msg in feedback_details:
                        cv2.putText(display_frame, f"👉 {msg}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                        y_offset += 40
            else:
                status_msg.warning("⚠️ Pose lost. Align your body in profile view flat on your back...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass