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
st.subheader("📸 AP Elbow Live Positioner (Fully Extended)")

# Inline placeholders for the video frame and system messages
frame_window = st.empty()  
status_msg = st.empty()

# ==========================================
# 1. HARDWARE THREADING LOCK (Zero Lag Buffer)
# ==========================================
class LiveVideoStream:
    def __init__(self, src=2):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.started = False

    def start(self):
        self.started = True
        return self

    def read(self):
        if not self.started: return None
        grabbed, frame = self.stream.read()
        if grabbed and frame is not None:
            return frame
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
    """Calculates the extension angle at the elbow vertex (point b)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Replaced cv2 trackbar with native Streamlit sidebar selection
cam_choice = st.sidebar.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2, 3], index=0)

# ==========================================
# 3. STREAMLIT SESSION STATE MANAGER
# ==========================================
if "active_straight_camera" not in st.session_state:
    st.session_state.active_straight_camera = None
if "current_straight_cam_idx" not in st.session_state:
    st.session_state.current_straight_cam_idx = -1

# Protects camera initialization states from memory stack leakage during page re-runs
if st.session_state.current_straight_cam_idx != cam_choice:
    if st.session_state.active_straight_camera is not None:
        st.session_state.active_straight_camera.stop()
    st.session_state.active_straight_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_straight_cam_idx = cam_choice
    time.sleep(0.5)

vs = st.session_state.active_straight_camera

# ==========================================
# 4. CONFIGURATION & MODELS
# ==========================================
BOX_SIZE = 250  
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")

pose_stabilizer = PoseStabilizer(alpha=0.28)

# ==========================================
# 5. INITIALIZE MEDIAPIPE VISION TASK
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.25,
    min_tracking_confidence=0.45
)
detector = vision.PoseLandmarker.create_from_options(options) 

# ==========================================
# 5. MAIN PROCESSING PIPELINE
# ==========================================
try:
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

        status_text = "PLACE EXTENDED AP ARM IN WORKSPACE"
        color = (0, 0, 255)
        side_label = "SCANNING..."
        feedback_details = []

        if result.pose_landmarks:
            smoothed_landmarks = pose_stabilizer.smooth(result.pose_landmarks[0])

            # Select the main visible arm
            left_score = smoothed_landmarks[11].visibility + smoothed_landmarks[13].visibility + smoothed_landmarks[15].visibility
            right_score = smoothed_landmarks[12].visibility + smoothed_landmarks[14].visibility + smoothed_landmarks[16].visibility

            if left_score > right_score:
                s_lm, e_lm, w_lm = smoothed_landmarks[11], smoothed_landmarks[13], smoothed_landmarks[15]
                pinky_lm, thumb_lm = smoothed_landmarks[17], smoothed_landmarks[21]
                side_label = "STANDARD AP: LEFT ELBOW"
            else:
                s_lm, e_lm, w_lm = smoothed_landmarks[12], smoothed_landmarks[14], smoothed_landmarks[16]
                pinky_lm, thumb_lm = smoothed_landmarks[18], smoothed_landmarks[22]
                side_label = "STANDARD AP: RIGHT ELBOW"

            # Translate landmarks into pixel coordinates
            shoulder_pt = (int((1 - s_lm.x) * w), int(s_lm.y * h))
            elbow_pt = (int((1 - e_lm.x) * w), int(e_lm.y * h))
            wrist_pt = (int((1 - w_lm.x) * w), int(w_lm.y * h))

            # Center the dynamic tracking box directly over the elbow joint
            target_x = elbow_pt[0] - (BOX_SIZE // 2)
            target_y = elbow_pt[1] - (BOX_SIZE // 2)

            # Calculate real-time arm extension angle
            extension_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)

            # ==========================================
            # CLINICAL AP CRITERIA VALIDATION
            # ==========================================
            extension_ok = 155.0 <= extension_angle <= 205.0
            hand_supinated = (thumb_lm.y < w_lm.y + 0.15) or (thumb_lm.y < pinky_lm.y + 0.15)
            
            se_dx = elbow_pt[0] - shoulder_pt[0]
            se_dy = elbow_pt[1] - shoulder_pt[1]
            se_tilt = np.degrees(np.arctan2(abs(se_dy), abs(se_dx) + 1e-6))

            ew_dx = wrist_pt[0] - elbow_pt[0]
            ew_dy = elbow_pt[1] - elbow_pt[1]
            ew_tilt = np.degrees(np.arctan2(abs(ew_dy), abs(ew_dx) + 1e-6))

            arm_horizontal = se_tilt <= 8.0 and ew_tilt <= 8.0

            good_posture = extension_ok and hand_supinated and arm_horizontal
            
            if good_posture:
                status_text = "RIGHT POSTURE"
                color = (0, 255, 0) # Green box lock
                status_msg.success(f"✅ POSITION PERFECT: AP Elbow Straight Line Locked ({extension_angle:.1f}°)")
            else:
                status_text = "WRONG POSTURE"
                color = (0, 0, 255) # Red warning box
                if not extension_ok or not arm_horizontal: 
                    feedback_details.append("Keep arm 180 degree horizontal")
                elif not hand_supinated: 
                    feedback_details.append("Turn Palm Face UP")
                status_msg.error(f"❌ Fix: {' | '.join(feedback_details)}")

            if 'global_telemetry' in globals():
                if good_posture:
                    global_telemetry['status'] = "good"
                    global_telemetry['message'] = "Perfect alignment. Keep holding."
                    global_telemetry['accuracy'] = 95
                else:
                    global_telemetry['status'] = "bad"
                    global_telemetry['accuracy'] = 45
                    if not extension_ok or not arm_horizontal:
                        global_telemetry['message'] = "Keep arm 180 degree horizontal"
                    elif not hand_supinated:
                        global_telemetry['message'] = "Turn Palm Face UP"
                    else:
                        global_telemetry['message'] = "Align your arm in camera view"

            # Draw the box following the straight joint space
            cv2.rectangle(display_frame, (target_x, target_y), (target_x + BOX_SIZE, target_y + BOX_SIZE), color, 2, cv2.LINE_AA)

            # Central Ray Axis Crosshair at the center of the joint space
            cv2.line(display_frame, (elbow_pt[0] - 40, elbow_pt[1]), (elbow_pt[0] + 40, elbow_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (elbow_pt[0], elbow_pt[1] - 40), (elbow_pt[0], elbow_pt[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Stabilized Skeleton Vector Paths
            cv2.line(display_frame, shoulder_pt, elbow_pt, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.circle(display_frame, shoulder_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(display_frame, elbow_pt, 8, color, -1, cv2.LINE_AA)   
            cv2.circle(display_frame, wrist_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

            # OSD Text Overlays
            cv2.putText(display_frame, side_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
        else:
            status_msg.warning("⚠️ Looking for active arm alignment in camera field of view...")
            cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            if 'global_telemetry' in globals():
                global_telemetry['status'] = "calibrating"
                global_telemetry['message'] = "Align your arm in camera view"
                global_telemetry['accuracy'] = 10

        # Convert back to RGB format and render inside the Streamlit web element layout panel
        rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(rgb_out, channels="RGB")

finally:
    detector.close()