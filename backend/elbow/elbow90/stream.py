import os
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STREAMLIT UI INLINE INITIALIZATION ---
st.title("🦾 Axoris Physio AI")
st.subheader("📸 Lateral Elbow Live Positioner (90° Flexion Profile)")

# Interactive drop-down choices widget to hot-swap cameras safely
cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

# Inline placeholders for browser rendering layout elements
frame_window = st.empty()  
status_msg = st.empty()

# ==========================================
# 1. HARDWARE THREADING LOCK (Zero Lag Buffer)
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
            time.sleep(0.01) # Keeps background processor thread overhead light

    def read(self):
        with self.read_lock:
            if self.grabbed and self.frame is not None: 
                return self.frame.copy()
            return None

    def stop(self):
        self.started = False
        if self.stream.isOpened(): self.stream.release()

# ==========================================
# 2. RAW COORDINATE FILTER (Prevents Jitter)
# ==========================================
class PoseStabilizer:
    def __init__(self, alpha=0.30): 
        self.alpha = alpha
        self.prev_l = None

    def smooth(self, current_l):
        if self.prev_l is None:
            self.prev_l = current_l
            return current_l
        smoothed = []
        for p, c in zip(self.prev_l, current_l):
            s_pt = type(c)(
                x = p.x * (1 - self.alpha) + c.x * self.alpha,
                y = p.y * (1 - self.alpha) + c.y * self.alpha,
                z = p.z * (1 - self.alpha) + c.z * self.alpha,
                visibility = c.visibility
            )
            smoothed.append(s_pt)
        self.prev_l = smoothed
        return smoothed

def calculate_angle_3pt(a, b, c):
    """Calculates the angle at point b (Elbow joint flexion angle)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 3. DYNAMIC REFRESH PERSISTENCE ENGINE
# ==========================================
if "active_lat_camera" not in st.session_state:
    st.session_state.active_lat_camera = None
if "current_lat_cam_idx" not in st.session_state:
    st.session_state.current_lat_cam_idx = -1

# Hot-reload camera parameters if selectbox is updated
if st.session_state.current_lat_cam_idx != cam_choice:
    if st.session_state.active_lat_camera is not None:
        st.session_state.active_lat_camera.stop()
    st.session_state.active_lat_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_lat_cam_idx = cam_choice

# ==========================================
# 4. CONFIGURATION & MODELS
# ==========================================
BOX_SIZE = 240  
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
pose_stabilizer = PoseStabilizer(alpha=0.28)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.25,
    min_tracking_confidence=0.45
)
detector = vision.PoseLandmarker.create_from_options(options)

vs = st.session_state.active_lat_camera

# ==========================================
# 5. MAIN PROCESSING PIPELINE
# ==========================================
try:
    if vs.frame is None:
        status_msg.error(f"❌ Camera Index {cam_choice} could not be initialized. Verify connection configurations.")
    else:
        # Loop continues executing as long as current tracking panel module is active
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

            status_text = "PLACE ARM IN WORKSPACE"
            color = (0, 0, 255)
            side_label = "SCANNING..."
            feedback_details = []

            if result.pose_landmarks:
                smoothed_landmarks = pose_stabilizer.smooth(result.pose_landmarks[0])

                # Select the main visible arm tracking structure details
                left_score = smoothed_landmarks[11].visibility + smoothed_landmarks[13].visibility + smoothed_landmarks[15].visibility
                right_score = smoothed_landmarks[12].visibility + smoothed_landmarks[14].visibility + smoothed_landmarks[16].visibility

                if left_score > right_score:
                    s_lm, e_lm, w_lm = smoothed_landmarks[11], smoothed_landmarks[13], smoothed_landmarks[15]
                    side_label = "LATERAL LEFT ELBOW"
                else:
                    s_lm, e_lm, w_lm = smoothed_landmarks[12], smoothed_landmarks[14], smoothed_landmarks[16]
                    side_label = "LATERAL RIGHT ELBOW"

                # Translate to matrix layout pixel units
                shoulder_pt = (int((1 - s_lm.x) * w), int(s_lm.y * h))
                elbow_pt = (int((1 - e_lm.x) * w), int(e_lm.y * h))
                wrist_pt = (int((1 - w_lm.x) * w), int(w_lm.y * h))

                # Center the box bounds directly on tracked elbow node point
                target_x = elbow_pt[0] - (BOX_SIZE // 2)
                target_y = elbow_pt[1] - (BOX_SIZE // 2)

                # Calculate clinical profile flexion angle
                flexion_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)
                forearm_len = np.linalg.norm(np.array(elbow_pt) - np.array(wrist_pt))

                # ==========================================
                # CLINICAL CRITERIA VALIDATION ROWS
                # ==========================================
                angle_ok = 80.0 <= flexion_angle <= 100.0
                extension_ok = 110.0 < forearm_len < 460.0

                good_posture = angle_ok and extension_ok
                
                if good_posture:
                    status_text = "RIGHT POSTURE"
                    color = (0, 255, 0) # Green box lock
                    status_msg.success(f"✅ POSITION PERFECT: True Lateral Profile Locked ({flexion_angle:.1f}°)")
                else:
                    status_text = "WRONG POSTURE"
                    color = (0, 0, 255) # Red warning box
                    if not angle_ok: 
                        feedback_details.append(f"Flex arm closer to 90° ({flexion_angle:.1f}° found)")
                    if not extension_ok: 
                        feedback_details.append("Keep upper arm and forearm flat against the surface plate profile")
                    status_msg.error(f"❌ Fix: {' | '.join(feedback_details)}")

                if 'global_telemetry' in globals():
                    if good_posture:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect alignment. Keep holding."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        if not angle_ok:
                            global_telemetry['message'] = "Flex arm closer to 90 degrees"
                        elif not extension_ok:
                            global_telemetry['message'] = "Keep upper arm and forearm flat against the surface plate"
                        else:
                            global_telemetry['message'] = "Align your arm in camera view"

                # Draw tracking frame
                cv2.rectangle(display_frame, (target_x, target_y), (target_x + BOX_SIZE, target_y + BOX_SIZE), color, 2, cv2.LINE_AA)

                # Crosshair target rendering matching configuration layout spec sheets
                cv2.line(display_frame, (elbow_pt[0] - 40, elbow_pt[1]), (elbow_pt[0] + 40, elbow_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(display_frame, (elbow_pt[0], elbow_pt[1] - 40), (elbow_pt[0], elbow_pt[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)

                # Draw skeletal telemetry links
                cv2.line(display_frame, shoulder_pt, elbow_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
                
                cv2.circle(display_frame, shoulder_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, elbow_pt, 8, color, -1, cv2.LINE_AA)   
                cv2.circle(display_frame, wrist_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

                # Text OSD layers properties
                cv2.putText(display_frame, side_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                status_msg.warning("⚠️ Scanning for stable arm layout profile inside workspace bounds...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "calibrating"
                    global_telemetry['message'] = "Align your arm in camera view"
                    global_telemetry['accuracy'] = 10

            # Output processed image arrays context into Streamlit layout framework
            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    detector.close()