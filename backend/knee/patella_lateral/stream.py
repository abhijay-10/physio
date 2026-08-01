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

def calculate_angle_3pt(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- STREAMLIT INLINE WORKSPACE INITIALIZATION ---
st.subheader("📸 Patella Lateral Live Positioner")
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

# ==========================================
# 2. TEMPORAL HISTORY BUFFER FILTER (Anti-Flicker & Lag-Free EMA)
# ==========================================
class FallbackBufferStabilizer:
    def __init__(self, alpha_high=0.3, alpha_low=0.15, window_size=5):
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.last_hip = None
        self.last_knee = None
        self.last_ankle = None
        self.last_box = None

    def process(self, hip, knee, ankle, hip_vis, knee_vis, ankle_vis):
        """EMA based dynamic tracker to eliminate coordinate latency and freeze anomalies"""
        # Hip tracking stabilizer
        if self.last_hip is None:
            self.last_hip = hip
        else:
            if hip_vis > 0.50:
                alpha = self.alpha_high
            elif hip_vis > 0.25:
                alpha = self.alpha_low
            else:
                alpha = 0.1
            self.last_hip = (
                int(alpha * hip[0] + (1 - alpha) * self.last_hip[0]),
                int(alpha * hip[1] + (1 - alpha) * self.last_hip[1])
            )

        # Knee tracking stabilizer
        if self.last_knee is None:
            self.last_knee = knee
        else:
            if knee_vis > 0.50:
                alpha = self.alpha_high
            elif knee_vis > 0.25:
                alpha = self.alpha_low
            else:
                alpha = 0.1
            self.last_knee = (
                int(alpha * knee[0] + (1 - alpha) * self.last_knee[0]),
                int(alpha * knee[1] + (1 - alpha) * self.last_knee[1])
            )

        # Ankle tracking stabilizer
        if self.last_ankle is None:
            self.last_ankle = ankle
        else:
            if ankle_vis > 0.50:
                alpha = self.alpha_high
            elif ankle_vis > 0.25:
                alpha = self.alpha_low
            else:
                alpha = 0.1
            self.last_ankle = (
                int(alpha * ankle[0] + (1 - alpha) * self.last_ankle[0]),
                int(alpha * ankle[1] + (1 - alpha) * self.last_ankle[1])
            )

        return self.last_hip, self.last_knee, self.last_ankle

    def smooth_box(self, current_corner):
        if self.last_box is None:
            self.last_box = current_corner
        else:
            # Responsive box center smoothing (alpha = 0.1)
            self.last_box = (
                int(0.1 * current_corner[0] + 0.9 * self.last_box[0]),
                int(0.1 * current_corner[1] + 0.9 * self.last_box[1])
            )
        return self.last_box

# Session state continuity checks
cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

if "active_hughston_camera" not in st.session_state:
    st.session_state.active_hughston_camera = None
if "current_hughston_cam_idx" not in st.session_state:
    st.session_state.current_hughston_cam_idx = -1

if st.session_state.current_hughston_cam_idx != cam_choice:
    if st.session_state.active_hughston_camera is not None:
        st.session_state.active_hughston_camera.stop()
    st.session_state.active_hughston_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_hughston_cam_idx = cam_choice

if "patella_angle_hist" not in st.session_state:
    st.session_state.patella_angle_hist = deque(maxlen=6)

# ==========================================
# 3. CONFIGURE RUNTIME PIPELINE
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
fallback_engine = FallbackBufferStabilizer()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.15, min_tracking_confidence=0.30        
)
detector = vision.PoseLandmarker.create_from_options(options)

vs = st.session_state.active_hughston_camera

# ==========================================
# 4. EXECUTION PROCESSING LOOP
# ==========================================
try:
    if vs.frame is None:
        st.error("❌ External Camera Index 2 could not be found. Check your USB port connections.")
    else:
        frame_count = 0
        last_landmarks = None
        consecutive_drops = 0
        last_timestamp_ms = 0

        while vs.started:
            # Yield CPU execution slice if camera hasn't captured a new frame
            if not vs.has_new_frame():
                time.sleep(0.002)
                continue

            frame = vs.read()
            if frame is None: continue

            frame_count += 1
            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape

            # Run Pose Landmarker inference on every frame to maintain continuous tracking and prevent flickering
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            
            # Ensure strictly increasing timestamp for MediaPipe Video mode (min 1ms delta)
            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            
            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            
            if result.pose_landmarks:
                last_landmarks = result.pose_landmarks[0]
                consecutive_drops = 0
            else:
                consecutive_drops += 1
                # Hard reset cached pose landmarks after 15 consecutive failed detections (~500ms)
                if consecutive_drops > 15:
                    last_landmarks = None

            status_text = "PLACE PATIENT IN PRONE CONFIGURATION"
            color = (0, 0, 255)
            side_label = "SCANNING PATELLOFEMORAL REGION..."
            feedback_details = []

            if last_landmarks is not None:
                landmarks = last_landmarks

                target_leg = "Auto"
                if hasattr(st.session_state, 'target_leg') and st.session_state.target_leg:
                    target_leg = st.session_state.target_leg
                
                if target_leg == "Left":
                    hip_lm, knee_lm, ankle_lm = landmarks[23], landmarks[25], landmarks[27]
                    side_label = "PATELLA LATERAL: LEFT LOWER LIMB"
                elif target_leg == "Right":
                    hip_lm, knee_lm, ankle_lm = landmarks[24], landmarks[26], landmarks[28]
                    side_label = "PATELLA LATERAL: RIGHT LOWER LIMB"
                else:
                    # Identify operational leg by visibility threshold scores
                    left_score = landmarks[23].visibility + landmarks[25].visibility + landmarks[27].visibility
                    right_score = landmarks[24].visibility + landmarks[26].visibility + landmarks[28].visibility

                    if left_score > right_score:
                        hip_lm, knee_lm, ankle_lm = landmarks[23], landmarks[25], landmarks[27]
                        side_label = "PATELLA LATERAL: LEFT LOWER LIMB"
                    else:
                        hip_lm, knee_lm, ankle_lm = landmarks[24], landmarks[26], landmarks[28]
                        side_label = "PATELLA LATERAL: RIGHT LOWER LIMB"

                # Translate normalized values to pixel positions
                raw_hip = (int((1 - hip_lm.x) * w), int(hip_lm.y * h))
                raw_knee = (int((1 - knee_lm.x) * w), int(knee_lm.y * h))
                raw_ankle = (int((1 - ankle_lm.x) * w), int(ankle_lm.y * h))

                # Boundary Check (Alert user if landmarks are cut off near the frame borders)
                edge_limit = 20
                out_of_bounds = False
                out_of_bounds_msg = ""
                if raw_ankle[0] <= edge_limit or raw_ankle[0] >= w - edge_limit or raw_ankle[1] <= edge_limit or raw_ankle[1] >= h - edge_limit:
                    out_of_bounds = True
                    out_of_bounds_msg = "Foot/Ankle out of frame"
                elif raw_knee[0] <= edge_limit or raw_knee[0] >= w - edge_limit or raw_knee[1] <= edge_limit or raw_knee[1] >= h - edge_limit:
                    out_of_bounds = True
                    out_of_bounds_msg = "Knee out of frame"
                elif raw_hip[0] <= edge_limit or raw_hip[0] >= w - edge_limit or raw_hip[1] <= edge_limit or raw_hip[1] >= h - edge_limit:
                    out_of_bounds = True
                    out_of_bounds_msg = "Hip out of frame"

                # Visibility Check (Alert user if landmarks are occluded/low confidence)
                visibility_ok = (hip_lm.visibility >= 0.25 and knee_lm.visibility >= 0.25 and ankle_lm.visibility >= 0.25)

                # Filter coordinates through history buffer to absorb dropouts
                hip_pt, knee_pt, ankle_pt = fallback_engine.process(
                    raw_hip, raw_knee, raw_ankle,
                    hip_lm.visibility, knee_lm.visibility, ankle_lm.visibility
                )

                # Dynamic sizing based on anatomical pixel dimension metrics
                pixel_scale = np.linalg.norm(np.array(knee_pt) - np.array(ankle_pt))
                dynamic_box_size = int(np.clip(pixel_scale * 0.70, 200, 340))

                # Smooth out bounding center transitions to extinguish frame jitter
                t_x = knee_pt[0] - (dynamic_box_size // 2)
                t_y = knee_pt[1] - (dynamic_box_size // 2)
                target_x, target_y = fallback_engine.smooth_box((t_x, t_y))

                # Bound limits clip protection layer
                target_x = np.clip(target_x, 10, w - dynamic_box_size - 10)
                target_y = np.clip(target_y, 10, h - dynamic_box_size - 10)

                # ==========================================
                # ✅ PATELLA LATERAL GEOMETRIC ANALYSIS LAW
                # ==========================================
                
                # Rule 1: Knee Flexion Angle Check (Knee joint must be bent to match lateral pose)
                raw_knee_angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
                st.session_state.patella_angle_hist.append(raw_knee_angle)
                knee_joint_angle = sum(st.session_state.patella_angle_hist) / len(st.session_state.patella_angle_hist)
                
                angle_ok = 75.0 <= knee_joint_angle <= 115.0

                # Rule 2: Knee Raised Check (Knee must be physically higher than the ankle - smaller Y value)
                knee_raised = knee_pt[1] < (ankle_pt[1] - (pixel_scale * 0.15))

                good_posture = angle_ok and knee_raised

                if out_of_bounds:
                    status_text = "ADJUST CAMERA"
                    color = (0, 165, 255)  # Orange warning color
                    status_msg.warning(f"⚠️ {out_of_bounds_msg.upper()}: Ensure your entire leg (hip to ankle) is visible within the screen.")
                else:
                    if good_posture:
                        status_text = "RIGHT POSTURE"
                        color = (0, 255, 0)
                        status_msg.success(f"✅ POSITION PERFECT: Patella Lateral Angle at {knee_joint_angle:.1f}° (Target: 75°-115°)")
                    else:
                        status_text = "WRONG POSTURE"
                        color = (0, 0, 255)
                        if not knee_raised:
                            feedback_details.append("Raise your knee up (knee should be higher than ankle)")
                        elif not angle_ok:
                            if knee_joint_angle > 115.0:
                                feedback_details.append(f"Bend your knee more ({knee_joint_angle:.1f}° found, target 75°-115°)")
                            else:
                                feedback_details.append(f"Straighten your knee slightly ({knee_joint_angle:.1f}° found, target 75°-115°)")
                        status_msg.error(f"❌ Fix: {' | '.join(feedback_details)}")

                # --- TELEMETRY updates for voice assistance ---
                if 'global_telemetry' in globals():
                    if out_of_bounds:
                        global_telemetry['status'] = "bad"
                        global_telemetry['message'] = f"Warning: {out_of_bounds_msg}"
                        global_telemetry['accuracy'] = 45
                    elif good_posture:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect alignment. Keep holding."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        if not knee_raised:
                            global_telemetry['message'] = "Warning: Raise your knee up"
                        else:
                            if knee_joint_angle > 115.0:
                                global_telemetry['message'] = "Warning: Bend your knee more"
                            else:
                                global_telemetry['message'] = "Warning: Straighten your knee slightly"

                # Draw smooth tangential collimation box
                cv2.rectangle(display_frame, (target_x, target_y), (target_x + dynamic_box_size, target_y + dynamic_box_size), color, 2, cv2.LINE_AA)
                
                # 45-Degree Cephalad Central Ray path simulation crosshair indicator matching reference layout
                cv2.line(display_frame, (knee_pt[0] - 50, knee_pt[1]), (knee_pt[0] + 50, knee_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(display_frame, (knee_pt[0], knee_pt[1] - 50), (knee_pt[0], knee_pt[1] + 50), (255, 255, 255), 1, cv2.LINE_AA)

                # Draw tracking vector connections lines
                cv2.line(display_frame, hip_pt, knee_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(display_frame, knee_pt, ankle_pt, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(display_frame, hip_pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(display_frame, knee_pt, 7, color, -1, cv2.LINE_AA)   
                cv2.circle(display_frame, ankle_pt, 5, (0, 255, 255), -1, cv2.LINE_AA)

                cv2.putText(display_frame, side_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                status_msg.warning("⚠️ Pose fields dropped. Center the prone knee profile inside camera layout bounds...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # --- TELEMETRY updates for voice assistance ---
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "calibrating"
                    global_telemetry['message'] = "Warning: Pose fields dropped. Center your knee in camera view."
                    global_telemetry['accuracy'] = 10

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass