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
st.subheader("📸 PA Axial Elbow Assistant (Coyle Method)")

# Replaced cv2 trackbar with native Streamlit sidebar selection
mode_select = st.sidebar.radio(
    "🎯 Select Projection Target Mode:",
    options=["Distal Humerus Focus (75° Flexion)", "Olecranon Process Focus (45° Flexion)"],
    index=0
)

# Interactive camera source selector to prevent hardware connection dropouts
cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

# Workspace window elements
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
            time.sleep(0.01) # Maintains ultra-low CPU load balancing

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
    """Calculates inner elbow joint flexion angle at vertex b"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 3. STREAMLIT SESSION STATE MANAGER
# ==========================================
if "active_axial_camera" not in st.session_state:
    st.session_state.active_axial_camera = None
if "current_axial_cam_idx" not in st.session_state:
    st.session_state.current_axial_cam_idx = -1

# Protects camera initialization states from memory stack leakage during page re-runs
if st.session_state.current_axial_cam_idx != cam_choice:
    if st.session_state.active_axial_camera is not None:
        st.session_state.active_axial_camera.stop()
    st.session_state.active_axial_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_axial_cam_idx = cam_choice

# ==========================================
# 4. CONFIGURATION & MODELS
# ==========================================
BOX_SIZE = 260  
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

vs = st.session_state.active_axial_camera

# ==========================================
# 5. MAIN PROCESSING PIPELINE
# ==========================================
try:
    if vs.frame is None:
        st.error(f"❌ Camera Source Index {cam_choice} could not be resolved. Change device port index layout above.")
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

            status_text = "PLACE ELBOW OBLIQUE FLAT ON PLATE"
            color = (0, 0, 255)
            side_text = "SCANNING REGION..."
            feedback_details = []

            if result.pose_landmarks:
                smoothed_landmarks = pose_stabilizer.smooth(result.pose_landmarks[0])

                # Split data array scores between right and left limb profiles
                left_score = smoothed_landmarks[11].visibility + smoothed_landmarks[13].visibility + smoothed_landmarks[15].visibility
                right_score = smoothed_landmarks[12].visibility + smoothed_landmarks[14].visibility + smoothed_landmarks[16].visibility

                if left_score > right_score:
                    s_lm, e_lm, w_lm = smoothed_landmarks[11], smoothed_landmarks[13], smoothed_landmarks[15]
                    pinky_lm, thumb_lm = smoothed_landmarks[17], smoothed_landmarks[21]
                    opp_w_lm = smoothed_landmarks[16]
                    side_text = "PA AXIAL: LEFT ELBOW"
                else:
                    s_lm, e_lm, w_lm = smoothed_landmarks[12], smoothed_landmarks[14], smoothed_landmarks[16]
                    pinky_lm, thumb_lm = smoothed_landmarks[18], smoothed_landmarks[22]
                    opp_w_lm = smoothed_landmarks[15]
                    side_text = "PA AXIAL: RIGHT ELBOW"

                # Map 2D output coordinates
                shoulder_pt = (int((1 - s_lm.x) * w), int(s_lm.y * h))
                elbow_pt = (int((1 - e_lm.x) * w), int(e_lm.y * h))
                wrist_pt = (int((1 - w_lm.x) * w), int(w_lm.y * h))

                target_x = elbow_pt[0] - (BOX_SIZE // 2)
                target_y = elbow_pt[1] - (BOX_SIZE // 2)

                flexion_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)

                # ==========================================
                # CLINICAL CRITERIA EVALUATION ROADMAP
                # ==========================================
                if mode_select == "Distal Humerus Focus (75° Flexion)":
                    angle_ok = 70.0 <= flexion_angle <= 81.0
                    target_string = f"Distal Humerus Angle: 75° ({flexion_angle:.1f}° Found)"
                    guide_msg = "👉 Move your hand OUTWARDS to open up the angle closer to 75°"
                else:
                    angle_ok = 40.0 <= flexion_angle <= 51.0
                    target_string = f"Olecranon Angle: 45° ({flexion_angle:.1f}° Found)"
                    guide_msg = "👉 Move your hand INWARDS / Lean lower to close angle to 45°"

                # Check hand position and cross-arm stabilization distance rules
                hand_supinated = thumb_lm.y < w_lm.y 
                immobilized_distance = np.linalg.norm(np.array([w_lm.x - opp_w_lm.x, w_lm.y - opp_w_lm.y]))
                is_immobilized = immobilized_distance < 0.22 
                stabilization_ok = hand_supinated and is_immobilized

                good_posture = angle_ok and stabilization_ok
                
                if good_posture:
                    status_text = "RIGHT POSTURE"
                    color = (0, 255, 0)
                    status_msg.success(f"✅ POSITION LOCK SECURED: Ready for Exposure ({flexion_angle:.1f}°)")
                else:
                    status_text = "WRONG POSTURE"
                    color = (0, 0, 255)
                    if not angle_ok: feedback_details.append(guide_msg)
                    if not stabilization_ok: feedback_details.append("👉 Open your palm facing CEILING and grab your wrist with your other hand.")
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
                            global_telemetry['message'] = guide_msg.replace("👉 ", "")
                        elif not stabilization_ok:
                            global_telemetry['message'] = "Open your palm facing ceiling and grab your wrist with your other hand"
                        else:
                            global_telemetry['message'] = "Align your arm in camera view"

                # Draw tracking frame bounding field boxes
                cv2.rectangle(display_frame, (target_x, target_y), (target_x + BOX_SIZE, target_y + BOX_SIZE), color, 2, cv2.LINE_AA)

                # Vector projection indicator arrows
                if mode_select == "Distal Humerus Focus (75° Flexion)":
                    cv2.arrowedLine(display_frame, (elbow_pt[0], elbow_pt[1] - 80), (elbow_pt[0], elbow_pt[1]), (255, 255, 255), 2, tipLength=0.2)
                else:
                    ray_start_x = elbow_pt[0] + int(80 * np.sin(np.radians(20)))
                    ray_start_y = elbow_pt[1] - int(80 * np.cos(np.radians(20)))
                    cv2.arrowedLine(display_frame, (ray_start_x, ray_start_y), (elbow_pt[0], elbow_pt[1]), (255, 255, 255), 2, tipLength=0.2)

                # Draw joint segments connection traces
                cv2.line(display_frame, shoulder_pt, elbow_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(display_frame, shoulder_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, elbow_pt, 8, color, -1, cv2.LINE_AA)   
                cv2.circle(display_frame, wrist_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

                cv2.putText(display_frame, side_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                status_msg.warning("⚠️ Waiting for arm presence alignment inside active scanning bounds...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "calibrating"
                    global_telemetry['message'] = "Align your arm in camera view"
                    global_telemetry['accuracy'] = 10

            # Route arrays directly into Streamlit view portal elements
            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    detector.close()