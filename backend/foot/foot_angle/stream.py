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
    """Calculates the interior angle at point b"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- STREAMLIT INLINE WORKSPACE INITIALIZATION ---
st.subheader("📸 Axoris Adaptive Posterior Leg Positioner")
st.info("🎥 Status: Optimized for Posterior Leg Alignment (Back view) and Perspective Calibration Locks.")

frame_window = st.empty()  
status_msg = st.empty()

# Fixed assignment context check after module imports
mp_image = mp.Image if hasattr(mp, 'Image') else None

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
# 2. PERSISTENT LEG TRACKING SMOOTHER
# ==========================================
class AdaptiveLegStabilizer:
    def __init__(self, window_size=12, box_alpha=0.05):
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

    def smooth_box(self, target_x, target_y, target_size):
        if self.prev_box_coords is None:
            self.prev_box_coords = (target_x, target_y, target_size)
            return target_x, target_y, target_size
        
        px, py, ps = self.prev_box_coords
        fx = int(px * (1 - self.box_alpha) + target_x * self.box_alpha)
        fy = int(py * (1 - self.box_alpha) + target_y * self.box_alpha)
        fs = int(ps * (1 - self.box_alpha) + target_size * self.box_alpha)
        
        self.prev_box_coords = (fx, fy, fs)
        return fx, fy, fs

# Manage state metrics across execution loops
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

# State variables for temporal hold / anti-flicker
if "last_landmarks" not in st.session_state:
    st.session_state.last_landmarks = None
if "pose_lost_counter" not in st.session_state:
    st.session_state.pose_lost_counter = 0

# Initialize history buffers for smoothing metrics over time (combating baggy clothing/pajama detection jumps)
for key in ["right_angles", "left_angles", "right_leans", "left_leans", "rotations", "tilts", "right_sags", "left_sags", "symmetry"]:
    session_key = f"hist_{key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = deque(maxlen=6)

# UI configuration selectbox
target_leg = st.sidebar.selectbox("🦵 Target Leg for Capture (Back)", ["Right Leg", "Left Leg", "Both Legs"])

# ==========================================
# 3. CONFIGURE MODEL GATES
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
engine_right = AdaptiveLegStabilizer(window_size=5, box_alpha=0.1)
engine_left = AdaptiveLegStabilizer(window_size=5, box_alpha=0.1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.15, min_tracking_confidence=0.35        
)
detector = vision.PoseLandmarker.create_from_options(options)
vs = st.session_state.active_foot_camera

# ==========================================
# 4. EXECUTION PROCESSING LOOP
# ==========================================
last_timestamp_ms = 0
try:
    if vs.frame is None:
        st.error("❌ External Camera Index 2 offline. Verify hardware cable links.")
    else:
        while vs.started:
            # Yield CPU execution slice if camera hasn't captured a new frame
            if not vs.has_new_frame():
                time.sleep(0.002)
                continue

            frame = vs.read()
            if frame is None: continue
            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape

            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            
            # Ensure strictly increasing timestamp
            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            try:
                result = detector.detect_for_video(mp_image_obj, current_timestamp_ms)
            except Exception as e:
                print(f"MediaPipe error caught: {e}")
                continue

            landmarks = None
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                st.session_state.last_landmarks = landmarks
                st.session_state.pose_lost_counter = 0
            else:
                # If pose is temporarily lost, use the cached landmarks up to 90 frames (approx 3 seconds at 30 FPS)
                if st.session_state.last_landmarks is not None and st.session_state.pose_lost_counter < 90:
                    landmarks = st.session_state.last_landmarks
                    st.session_state.pose_lost_counter += 1
                else:
                    st.session_state.last_landmarks = None

            status_text = "ALIGN LEG WITH CENTER AXIS"
            color = (0, 0, 255)
            feedback_details = []

            if landmarks is not None:
                # Check visibility for Right Leg (landmarks 23, 25, 27)
                right_leg_visible = (landmarks[23].visibility > 0.15 and 
                                     landmarks[25].visibility > 0.15 and 
                                     landmarks[27].visibility > 0.15)
                
                # Check visibility for Left Leg (landmarks 24, 26, 28)
                left_leg_visible = (landmarks[24].visibility > 0.15 and 
                                    landmarks[26].visibility > 0.15 and 
                                    landmarks[28].visibility > 0.15)

                # Initialize states for both legs
                right_state = {"visible": False, "angle": 180.0, "ok": False, "box": None, "angle_ok": True, "leaning_ok": True, "motion_ok": True}
                left_state = {"visible": False, "angle": 180.0, "ok": False, "box": None, "angle_ok": True, "leaning_ok": True, "motion_ok": True}

                if right_leg_visible:
                    raw_hip = (int((1 - landmarks[23].x) * w), int(landmarks[23].y * h))
                    raw_knee = (int((1 - landmarks[25].x) * w), int(landmarks[25].y * h))
                    raw_ankle = (int((1 - landmarks[27].x) * w), int(landmarks[27].y * h))
                    hip_pt, knee_pt, ankle_pt = engine_right.smooth(raw_hip, raw_knee, raw_ankle)
                    
                    angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
                    # Smooth raw knee angle over history
                    st.session_state["hist_right_angles"].append(angle)
                    smooth_angle = np.mean(st.session_state["hist_right_angles"])
                    angle_ok = 165.0 <= smooth_angle <= 180.0
                    
                    leaning_angle = np.degrees(np.arctan2(abs(ankle_pt[0] - hip_pt[0]), abs(ankle_pt[1] - hip_pt[1]) + 1e-6))
                    # Smooth leaning angle over history
                    st.session_state["hist_right_leans"].append(leaning_angle)
                    smooth_lean = np.mean(st.session_state["hist_right_leans"])
                    leaning_ok = smooth_lean <= 10.0 # strict check for any sideways lean
                    
                    # Motion check
                    motion_ok = True
                    prev_key = "prev_knee_right"
                    if prev_key in st.session_state and st.session_state[prev_key] is not None:
                        dist = np.linalg.norm(np.array(knee_pt) - np.array(st.session_state[prev_key]))
                        if dist > 8.0:
                            motion_ok = False
                    st.session_state[prev_key] = knee_pt
                    
                    # Bounding box
                    xs = [hip_pt[0], knee_pt[0], ankle_pt[0]]
                    ys = [hip_pt[1], knee_pt[1], ankle_pt[1]]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
                    span = max(max_x - min_x, max_y - min_y)
                    raw_box_size = int(np.clip(span + 100, 300, 700))
                    raw_tx = center_x - (raw_box_size // 2)
                    raw_ty = center_y - (raw_box_size // 2)
                    
                    tx, ty, dbs = engine_right.smooth_box(raw_tx, raw_ty, raw_box_size)
                    tx = np.clip(tx, 10, w - dbs - 10)
                    ty = np.clip(ty, 10, h - dbs - 10)
                    
                    right_state = {
                        "visible": True,
                        "angle": smooth_angle,
                        "raw_angle": angle,
                        "ok": angle_ok and leaning_ok and motion_ok,
                        "angle_ok": angle_ok,
                        "leaning_ok": leaning_ok,
                        "motion_ok": motion_ok,
                        "hip_pt": hip_pt,
                        "knee_pt": knee_pt,
                        "ankle_pt": ankle_pt,
                        "box": (tx, ty, dbs)
                    }

                if left_leg_visible:
                    raw_hip = (int((1 - landmarks[24].x) * w), int(landmarks[24].y * h))
                    raw_knee = (int((1 - landmarks[26].x) * w), int(landmarks[26].y * h))
                    raw_ankle = (int((1 - landmarks[28].x) * w), int(landmarks[28].y * h))
                    hip_pt, knee_pt, ankle_pt = engine_left.smooth(raw_hip, raw_knee, raw_ankle)
                    
                    angle = calculate_angle_3pt(hip_pt, knee_pt, ankle_pt)
                    # Smooth raw knee angle over history
                    st.session_state["hist_left_angles"].append(angle)
                    smooth_angle = np.mean(st.session_state["hist_left_angles"])
                    angle_ok = 165.0 <= smooth_angle <= 180.0
                    
                    leaning_angle = np.degrees(np.arctan2(abs(ankle_pt[0] - hip_pt[0]), abs(ankle_pt[1] - hip_pt[1]) + 1e-6))
                    # Smooth leaning angle over history
                    st.session_state["hist_left_leans"].append(leaning_angle)
                    smooth_lean = np.mean(st.session_state["hist_left_leans"])
                    leaning_ok = smooth_lean <= 10.0 # strict check for any sideways lean
                    
                    # Motion check
                    motion_ok = True
                    prev_key = "prev_knee_left"
                    if prev_key in st.session_state and st.session_state[prev_key] is not None:
                        dist = np.linalg.norm(np.array(knee_pt) - np.array(st.session_state[prev_key]))
                        if dist > 8.0:
                            motion_ok = False
                    st.session_state[prev_key] = knee_pt
                    
                    # Bounding box
                    xs = [hip_pt[0], knee_pt[0], ankle_pt[0]]
                    ys = [hip_pt[1], knee_pt[1], ankle_pt[1]]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
                    span = max(max_x - min_x, max_y - min_y)
                    raw_box_size = int(np.clip(span + 100, 300, 700))
                    raw_tx = center_x - (raw_box_size // 2)
                    raw_ty = center_y - (raw_box_size // 2)
                    
                    tx, ty, dbs = engine_left.smooth_box(raw_tx, raw_ty, raw_box_size)
                    tx = np.clip(tx, 10, w - dbs - 10)
                    ty = np.clip(ty, 10, h - dbs - 10)
                    
                    left_state = {
                        "visible": True,
                        "angle": smooth_angle,
                        "raw_angle": angle,
                        "ok": angle_ok and leaning_ok and motion_ok,
                        "angle_ok": angle_ok,
                        "leaning_ok": leaning_ok,
                        "motion_ok": motion_ok,
                        "hip_pt": hip_pt,
                        "knee_pt": knee_pt,
                        "ankle_pt": ankle_pt,
                        "box": (tx, ty, dbs)
                    }

                # Global post-processing/validation:
                both_visible = right_state["visible"] and left_state["visible"]
                
                # Default global statuses to True if they can't be computed
                rotation_ok = True
                facing_back_ok = True
                hip_tilt_ok = True
                right_sagittal_ok = True
                left_sagittal_ok = True
                
                rotation_angle = 0.0
                hip_tilt_angle = 0.0
                right_sagittal_lean = 0.0
                left_sagittal_lean = 0.0

                if both_visible:
                    # Body rotation check (hips rotation around Y axis) - strict check for turning sideways
                    dx = landmarks[23].x - landmarks[24].x
                    dz = landmarks[23].z - landmarks[24].z
                    raw_rotation = np.degrees(np.arctan2(abs(dz), abs(dx) + 1e-6))
                    st.session_state["hist_rotations"].append(raw_rotation)
                    rotation_angle = np.mean(st.session_state["hist_rotations"])
                    rotation_ok = rotation_angle <= 35.0
                    
                    # Robust facing away check (Left Hip X MUST be strictly to the left of Right Hip X in camera view)
                    facing_back_ok = landmarks[23].x < landmarks[24].x
                    
                    # Hip tilt check (pelvic tilt) - strict check
                    hip_dy = landmarks[23].y - landmarks[24].y
                    hip_dx = landmarks[23].x - landmarks[24].x
                    raw_hip_tilt = np.degrees(np.arctan2(abs(hip_dy), abs(hip_dx) + 1e-6))
                    st.session_state["hist_tilts"].append(raw_hip_tilt)
                    hip_tilt_angle = np.mean(st.session_state["hist_tilts"])
                    hip_tilt_ok = hip_tilt_angle <= 15.0
                    
                    # Sagittal leaning check - strict check to avoid leaning forward/backward
                    raw_right_sagittal = np.degrees(np.arctan2(abs(landmarks[24].z - landmarks[28].z), abs(landmarks[24].y - landmarks[28].y) + 1e-6))
                    raw_left_sagittal = np.degrees(np.arctan2(abs(landmarks[23].z - landmarks[27].z), abs(landmarks[23].y - landmarks[27].y) + 1e-6))
                    
                    st.session_state["hist_right_sags"].append(raw_right_sagittal)
                    st.session_state["hist_left_sags"].append(raw_left_sagittal)
                    
                    right_sagittal_lean = np.mean(st.session_state["hist_right_sags"])
                    left_sagittal_lean = np.mean(st.session_state["hist_left_sags"])
                    
                    right_sagittal_ok = right_sagittal_lean <= 35.0
                    left_sagittal_ok = left_sagittal_lean <= 35.0
                    
                    # Leg length symmetry check (detects forward bending from back view)
                    left_len = abs(landmarks[27].y - landmarks[23].y)
                    right_len = abs(landmarks[28].y - landmarks[24].y)
                    raw_sym = abs(left_len - right_len) / (max(left_len, right_len) + 1e-6)
                    st.session_state["hist_symmetry"].append(raw_sym)
                    symmetry_ratio = np.mean(st.session_state["hist_symmetry"])
                    symmetry_ok = symmetry_ratio <= 0.05

                right_angle_ok = right_state["angle_ok"]
                left_angle_ok = left_state["angle_ok"]
                
                right_leaning_ok = right_state["leaning_ok"]
                left_leaning_ok = left_state["leaning_ok"]

                global_ok = (
                    both_visible and
                    rotation_ok and
                    facing_back_ok and
                    hip_tilt_ok and
                    right_sagittal_ok and
                    left_sagittal_ok and
                    symmetry_ok and
                    right_angle_ok and
                    right_leaning_ok and
                    right_state["motion_ok"] and
                    left_angle_ok and
                    left_leaning_ok and
                    left_state["motion_ok"]
                )

                if right_state["visible"]:
                    right_state["ok"] = global_ok
                if left_state["visible"]:
                    left_state["ok"] = global_ok

                # Determine active validation status based on selected target leg
                final_ok = False
                out_of_bounds = False
                out_of_bounds_msg = ""

                if target_leg == "Right Leg":
                    if not both_visible:
                        out_of_bounds = True
                        if not right_state["visible"]:
                            out_of_bounds_msg = "Right Leg not visible"
                        else:
                            out_of_bounds_msg = "Left Leg not visible (both legs must be visible)"
                    else:
                        final_ok = right_state["ok"]
                        leg_angle = right_state["angle"]
                        if final_ok:
                            status_text = "RIGHT LEG: OK"
                            color = (0, 255, 0)
                            status_msg.success(f"✅ RIGHT LEG LOCKED: Back Alignment Validated ({leg_angle:.1f}°).")
                        else:
                            status_text = "RIGHT LEG: WRONG POSTURE"
                            color = (0, 0, 255)
                            if not right_angle_ok:
                                feedback_details.append(f"Avoid bending. Straighten right leg ({leg_angle:.1f}°).")
                            if not right_leaning_ok:
                                feedback_details.append("Avoid leaning/tilting. Stand vertically straight in front.")
                            if not right_state["motion_ok"]:
                                feedback_details.append("Avoid moving. Hold completely still.")
                            if not rotation_ok:
                                feedback_details.append(f"Avoid turning sideways ({rotation_angle:.1f}°/35.0°).")
                            if not facing_back_ok:
                                feedback_details.append("Avoid facing the camera. Turn to face away from the camera.")
                            if not hip_tilt_ok:
                                feedback_details.append(f"Avoid tilting hips ({hip_tilt_angle:.1f}°/15.0°).")
                            if not (right_sagittal_ok and left_sagittal_ok):
                                feedback_details.append(f"Avoid leaning forward or backward (R: {right_sagittal_lean:.1f}°, L: {left_sagittal_lean:.1f}°).")
                            if not symmetry_ok:
                                feedback_details.append("Both legs must be equally straight. Do not rest or bend one leg.")
                            if not left_angle_ok:
                                feedback_details.append(f"Straighten left leg too ({left_state['angle']:.1f}°).")
                            if not left_leaning_ok:
                                feedback_details.append("Keep left leg vertical.")
                            if not left_state["motion_ok"]:
                                feedback_details.append("Keep left leg still.")
                            status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")
                
                elif target_leg == "Left Leg":
                    if not both_visible:
                        out_of_bounds = True
                        if not left_state["visible"]:
                            out_of_bounds_msg = "Left Leg not visible"
                        else:
                            out_of_bounds_msg = "Right Leg not visible (both legs must be visible)"
                    else:
                        final_ok = left_state["ok"]
                        leg_angle = left_state["angle"]
                        if final_ok:
                            status_text = "LEFT LEG: OK"
                            color = (0, 255, 0)
                            status_msg.success(f"✅ LEFT LEG LOCKED: Back Alignment Validated ({leg_angle:.1f}°).")
                        else:
                            status_text = "LEFT LEG: WRONG POSTURE"
                            color = (0, 0, 255)
                            if not left_angle_ok:
                                feedback_details.append(f"Avoid bending. Straighten left leg ({leg_angle:.1f}°).")
                            if not left_leaning_ok:
                                feedback_details.append("Avoid leaning/tilting. Stand vertically straight in front.")
                            if not left_state["motion_ok"]:
                                feedback_details.append("Avoid moving. Hold completely still.")
                            if not rotation_ok:
                                feedback_details.append(f"Avoid turning sideways ({rotation_angle:.1f}°/35.0°).")
                            if not facing_back_ok:
                                feedback_details.append("Avoid facing the camera. Turn to face away from the camera.")
                            if not hip_tilt_ok:
                                feedback_details.append(f"Avoid tilting hips ({hip_tilt_angle:.1f}°/15.0°).")
                            if not (right_sagittal_ok and left_sagittal_ok):
                                feedback_details.append(f"Avoid leaning forward or backward (R: {right_sagittal_lean:.1f}°, L: {left_sagittal_lean:.1f}°).")
                            if not symmetry_ok:
                                feedback_details.append("Both legs must be equally straight. Do not rest or bend one leg.")
                            if not right_angle_ok:
                                feedback_details.append(f"Straighten right leg too ({right_state['angle']:.1f}°).")
                            if not right_leaning_ok:
                                feedback_details.append("Keep right leg vertical.")
                            if not right_state["motion_ok"]:
                                feedback_details.append("Keep right leg still.")
                            status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")
                
                else: # "Both Legs"
                    if not both_visible:
                        out_of_bounds = True
                        if not right_state["visible"] and not left_state["visible"]:
                            out_of_bounds_msg = "Both legs must be visible"
                        elif not right_state["visible"]:
                            out_of_bounds_msg = "Right Leg not visible (both legs must be visible)"
                        else:
                            out_of_bounds_msg = "Left Leg not visible (both legs must be visible)"
                    else:
                        final_ok = right_state["ok"] and left_state["ok"]
                        r_angle = right_state["angle"]
                        l_angle = left_state["angle"]
                        if final_ok:
                            status_text = "BOTH LEGS: OK"
                            color = (0, 255, 0)
                            status_msg.success(f"✅ BOTH LEGS LOCKED: Right ({r_angle:.1f}°), Left ({l_angle:.1f}°).")
                        else:
                            status_text = "BOTH LEGS: WRONG POSTURE"
                            color = (0, 0, 255)
                            if not right_state["angle_ok"]:
                                feedback_details.append(f"Straighten right leg (current: {r_angle:.1f}°)")
                            if not right_state["leaning_ok"]:
                                feedback_details.append("Right leg is leaning/tilted")
                            if not right_state["motion_ok"]:
                                feedback_details.append("Right leg is moving")
                            if not left_state["angle_ok"]:
                                feedback_details.append(f"Straighten left leg (current: {l_angle:.1f}°)")
                            if not left_state["leaning_ok"]:
                                feedback_details.append("Left leg is leaning/tilted")
                            if not left_state["motion_ok"]:
                                feedback_details.append("Left leg is moving")
                            if not rotation_ok:
                                feedback_details.append(f"Avoid turning sideways ({rotation_angle:.1f}°/35.0°).")
                            if not facing_back_ok:
                                feedback_details.append("Avoid facing the camera. Turn to face away from the camera.")
                            if not hip_tilt_ok:
                                feedback_details.append(f"Avoid tilting hips ({hip_tilt_angle:.1f}°/15.0°).")
                            if not (right_sagittal_ok and left_sagittal_ok):
                                feedback_details.append(f"Avoid leaning forward or backward (R: {right_sagittal_lean:.1f}°, L: {left_sagittal_lean:.1f}°).")
                            status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")

                if out_of_bounds:
                    status_text = "ADJUST CAMERA"
                    color = (0, 165, 255) # Orange warning
                    status_msg.warning(f"⚠️ {out_of_bounds_msg.upper()}: Ensure target leg is fully visible.")

                # --- TELEMETRY updates for voice assistance & API ---
                if 'global_telemetry' in globals():
                    if out_of_bounds:
                        global_telemetry['status'] = "bad"
                        global_telemetry['message'] = f"Warning: {out_of_bounds_msg}."
                        global_telemetry['accuracy'] = 45
                    elif final_ok:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect back alignment. Keep holding."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        global_telemetry['message'] = f"Warning: {' | '.join(feedback_details)}"

                # Draw overlays for Left Leg if visible
                if left_state["visible"]:
                    tx, ty, dbs = left_state["box"]
                    hip_pt, knee_pt, ankle_pt = left_state["hip_pt"], left_state["knee_pt"], left_state["ankle_pt"]
                    leg_color = color if (target_leg in ["Left Leg", "Both Legs"]) else (255, 255, 255)
                    
                    cv2.rectangle(display_frame, (tx, ty), (tx + dbs, ty + dbs), leg_color, 2, cv2.LINE_AA)
                    cv2.line(display_frame, (knee_pt[0] - 30, knee_pt[1]), (knee_pt[0] + 30, knee_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(display_frame, (knee_pt[0], knee_pt[1] - 30), (knee_pt[0], knee_pt[1] + 30), (255, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.line(display_frame, hip_pt, knee_pt, (255, 100, 100), 2, cv2.LINE_AA)
                    cv2.line(display_frame, knee_pt, ankle_pt, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.circle(display_frame, hip_pt, 8, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(display_frame, knee_pt, 10, leg_color, -1, cv2.LINE_AA)
                    cv2.circle(display_frame, ankle_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Left Knee: {int(left_state['angle'])} deg", (knee_pt[0] - 60, knee_pt[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw overlays for Right Leg if visible
                if right_state["visible"]:
                    tx, ty, dbs = right_state["box"]
                    hip_pt, knee_pt, ankle_pt = right_state["hip_pt"], right_state["knee_pt"], right_state["ankle_pt"]
                    leg_color = color if (target_leg in ["Right Leg", "Both Legs"]) else (255, 255, 255)
                    
                    cv2.rectangle(display_frame, (tx, ty), (tx + dbs, ty + dbs), leg_color, 2, cv2.LINE_AA)
                    cv2.line(display_frame, (knee_pt[0] - 30, knee_pt[1]), (knee_pt[0] + 30, knee_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(display_frame, (knee_pt[0], knee_pt[1] - 30), (knee_pt[0], knee_pt[1] + 30), (255, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.line(display_frame, hip_pt, knee_pt, (255, 100, 100), 2, cv2.LINE_AA)
                    cv2.line(display_frame, knee_pt, ankle_pt, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.circle(display_frame, hip_pt, 8, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(display_frame, knee_pt, 10, leg_color, -1, cv2.LINE_AA)
                    cv2.circle(display_frame, ankle_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Right Knee: {int(right_state['angle'])} deg", (knee_pt[0] - 60, knee_pt[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                # Overall HUD overlays
                cv2.putText(display_frame, f"TARGET: {target_leg.upper()} (BACK)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "bad"
                    global_telemetry['message'] = "Warning: Pose lost. Align your leg inside the camera window bounds..."
                    global_telemetry['accuracy'] = 0
                status_msg.warning("⚠️ Pose lost. Align your leg inside the camera window bounds...")
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass
