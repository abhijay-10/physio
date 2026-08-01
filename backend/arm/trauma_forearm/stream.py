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

st.subheader("📸 Axoris Back Arm Trauma Positioner")
st.info("🎥 Status: Auto-detecting the forearm resting straight.")

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

class AdaptiveForearmStabilizer:
    def __init__(self, box_alpha=0.05):
        self.box_alpha = box_alpha
        self.s_shoulder = None
        self.s_elbow = None
        self.s_wrist = None
        self.s_index = None
        self.prev_box_coords = None

    def smooth_pt(self, curr, target):
        if curr is None: return np.array(target, dtype=float)
        dist = math.hypot(target[0] - curr[0], target[1] - curr[1])
        if dist > 300: return np.array(target, dtype=float)
        
        alpha = np.clip(dist / 50.0, 0.05, 0.5)
        return curr * (1 - alpha) + np.array(target, dtype=float) * alpha

    def smooth(self, shoulder, elbow, wrist, index):
        self.s_shoulder = self.smooth_pt(self.s_shoulder, shoulder)
        self.s_elbow = self.smooth_pt(self.s_elbow, elbow)
        self.s_wrist = self.smooth_pt(self.s_wrist, wrist)
        self.s_index = self.smooth_pt(self.s_index, index)
        
        return (
            tuple(self.s_shoulder.astype(int)),
            tuple(self.s_elbow.astype(int)),
            tuple(self.s_wrist.astype(int)),
            tuple(self.s_index.astype(int))
        )

    def smooth_box(self, tx, ty, size):
        if self.prev_box_coords is None:
            self.prev_box_coords = np.array([tx, ty, size], dtype=float)
        else:
            curr = np.array([tx, ty, size], dtype=float)
            self.prev_box_coords = self.prev_box_coords * (1 - self.box_alpha) + curr * self.box_alpha
        out = self.prev_box_coords.astype(int)
        return out[0], out[1], out[2]

cam_choice = st.selectbox("🎥 Select Camera Input Source Device:", options=[0, 1, 2], index=0)

if "active_arm_trauma_camera" not in st.session_state:
    st.session_state.active_arm_trauma_camera = None

if "current_arm_trauma_cam_idx" not in st.session_state:
    st.session_state.current_arm_trauma_cam_idx = -1

if st.session_state.current_arm_trauma_cam_idx != cam_choice:
    if st.session_state.active_arm_trauma_camera is not None:
        st.session_state.active_arm_trauma_camera.stop()
    st.session_state.active_arm_trauma_camera = LiveVideoStream(src=cam_choice).start()
    st.session_state.current_arm_trauma_cam_idx = cam_choice

if "last_landmarks" not in st.session_state:
    st.session_state.last_landmarks = None
if "pose_lost_counter" not in st.session_state:
    st.session_state.pose_lost_counter = 0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
engine_right = AdaptiveForearmStabilizer(box_alpha=0.05)
engine_left = AdaptiveForearmStabilizer(box_alpha=0.05)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.6, 
    min_tracking_confidence=0.6,
    min_pose_presence_confidence=0.6
)
detector = vision.PoseLandmarker.create_from_options(options)
vs = st.session_state.active_arm_trauma_camera

def check_forearm_state(landmarks, h, w, shoulder_idx, elbow_idx, wrist_idx, index_idx, engine):
    raw_shoulder = (int((1 - landmarks[shoulder_idx].x) * w), int(landmarks[shoulder_idx].y * h))
    raw_elbow = (int((1 - landmarks[elbow_idx].x) * w), int(landmarks[elbow_idx].y * h))
    raw_wrist = (int((1 - landmarks[wrist_idx].x) * w), int(landmarks[wrist_idx].y * h))
    raw_index = (int((1 - landmarks[index_idx].x) * w), int(landmarks[index_idx].y * h))
    
    pixel_span = math.hypot(raw_index[0] - raw_elbow[0], raw_index[1] - raw_elbow[1])
    
    shoulder_pt, elbow_pt, wrist_pt, index_pt = engine.smooth(raw_shoulder, raw_elbow, raw_wrist, raw_index)
    
    def extrapolate(wrist, knuckle, factor):
        return (
            int(wrist[0] + (knuckle[0] - wrist[0]) * factor),
            int(wrist[1] + (knuckle[1] - wrist[1]) * factor)
        )
    tip_pt = extrapolate(wrist_pt, index_pt, 1.6)
    
    # 1. Hand straight (in line with forearm), flat, and palm facing down
    wrist_angle = calculate_angle_3pt(elbow_pt, wrist_pt, tip_pt)
    wrist_straight = wrist_angle > 155.0 
    
    forearm_len = math.hypot(wrist_pt[0] - elbow_pt[0], wrist_pt[1] - elbow_pt[1])
    hand_len = math.hypot(index_pt[0] - wrist_pt[0], index_pt[1] - wrist_pt[1])
    hand_ratio = hand_len / (forearm_len + 1e-6)
    flat_hand_ok = hand_ratio > 0.18
    
    # Check if palm is facing down (pronated). 
    # Since the user lies on their stomach with head towards the top (smaller Y) and arm resting on the bed (larger Y), 
    # a palm-down hand will have the thumb pointing towards the chest (smaller Y) and the pinky pointing away (larger Y).
    is_right = (wrist_idx == 16)
    thumb_idx = 22 if is_right else 21
    pinky_idx = 18 if is_right else 17
    
    raw_thumb = (int((1 - landmarks[thumb_idx].x) * w), int(landmarks[thumb_idx].y * h))
    raw_pinky = (int((1 - landmarks[pinky_idx].x) * w), int(landmarks[pinky_idx].y * h))
    
    palm_down = raw_pinky[1] > raw_thumb[1] + 5
    
    hand_ok = wrist_straight and flat_hand_ok and palm_down
    
    # 2. Forearm must be resting horizontally in camera view
    dy = abs(wrist_pt[1] - elbow_pt[1])
    dx = abs(wrist_pt[0] - elbow_pt[0])
    leaning_angle = np.degrees(np.arctan2(dy, dx + 1e-6))
    leaning_ok = leaning_angle <= 25.0
    
    # 3. Elbow must be bent at the correct angle (first image pose is ~100 deg)
    elbow_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)
    elbow_ok = 80.0 <= elbow_angle <= 102.0

    motion_ok = True
    prev_key = f"prev_elbow_{elbow_idx}"
    if prev_key in st.session_state and st.session_state[prev_key] is not None:
        dist = np.linalg.norm(np.array(elbow_pt) - np.array(st.session_state[prev_key]))
        if dist > 40.0:
            motion_ok = False
    st.session_state[prev_key] = elbow_pt
    
    xs = [elbow_pt[0], wrist_pt[0], tip_pt[0]]
    ys = [elbow_pt[1], wrist_pt[1], tip_pt[1]]
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
    
    avg_z = (landmarks[elbow_idx].z + landmarks[wrist_idx].z) / 2.0
    
    return {
        "visible": True,
        "span": pixel_span,
        "avg_z": avg_z,
        "wrist_angle": wrist_angle,
        "hand_ok": hand_ok,
        "flat_hand_ok": flat_hand_ok,
        "palm_down": palm_down,
        "leaning_ok": leaning_ok,
        "leaning_angle": leaning_angle,
        "elbow_ok": elbow_ok,
        "elbow_angle": elbow_angle,
        "motion_ok": motion_ok,
        "shoulder_pt": shoulder_pt,
        "elbow_pt": elbow_pt,
        "wrist_pt": wrist_pt,
        "index_pt": tip_pt,
        "box": (tx, ty, dbs)
    }

def draw_forearm(frame, state, color):
    if not state: return
    tx, ty, dbs = state["box"]
    shoulder_pt, elbow_pt, wrist_pt, index_pt = state["shoulder_pt"], state["elbow_pt"], state["wrist_pt"], state["index_pt"]
    
    cv2.rectangle(frame, (tx, ty), (tx + dbs, ty + dbs), color, 2, cv2.LINE_AA)
    
    # Upper arm
    cv2.line(frame, shoulder_pt, elbow_pt, (255, 100, 100), 2, cv2.LINE_AA)
    # Forearm
    cv2.line(frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Hand line
    cv2.line(frame, wrist_pt, index_pt, (100, 255, 100), 3, cv2.LINE_AA)
    
    # Joints
    cv2.circle(frame, shoulder_pt, 8, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, elbow_pt, 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, wrist_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, index_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Wrist: {int(state['wrist_angle'])} deg", (wrist_pt[0] + 20, wrist_pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tilt: {int(state['leaning_angle'])} deg", (wrist_pt[0] + 20, wrist_pt[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

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
                if st.session_state.last_landmarks is not None and st.session_state.pose_lost_counter < 5:
                    landmarks = st.session_state.last_landmarks
                    st.session_state.pose_lost_counter += 1
                else:
                    st.session_state.last_landmarks = None

            color = (0, 0, 255)
            status_text = "ALIGN FOREARM AND KEEP HAND STRAIGHT"
            feedback_details = []

            if landmarks is not None:
                # Right arm (shoulder=12, elbow=14, wrist=16, index=20)
                right_state = check_forearm_state(landmarks, h, w, 12, 14, 16, 20, engine_right)
                # Left arm (shoulder=11, elbow=13, wrist=15, index=19)
                left_state = check_forearm_state(landmarks, h, w, 11, 13, 15, 19, engine_left)

                primary_state = None
                if right_state and left_state:
                    if right_state["span"] > left_state["span"]:
                        primary_state = right_state
                    else:
                        primary_state = left_state
                elif right_state:
                    primary_state = right_state
                elif left_state:
                    primary_state = left_state

                if primary_state:
                    final_ok = primary_state["hand_ok"] and primary_state["leaning_ok"] and primary_state["elbow_ok"] and primary_state["motion_ok"]
                    
                    if final_ok:
                        status_text = "BACK ARM TRAUMA: OK"
                        color = (0, 255, 0)
                        status_msg.success(f"✅ Pose is correct. Arm is straight and flat.")
                    else:
                        status_text = "BACK ARM TRAUMA: ADJUST"
                        color = (0, 0, 255)
                        if not primary_state["hand_ok"]:
                            if not primary_state["flat_hand_ok"]:
                                feedback_details.append(f"Keep hand open and flat, do not curl fingers.")
                            elif not primary_state["palm_down"]:
                                feedback_details.append(f"Twist wrist so your palm faces down (back of hand faces up).")
                            else:
                                feedback_details.append(f"Keep hand straight with forearm.")
                        if not primary_state["elbow_ok"]:
                            feedback_details.append(f"Adjust elbow bend (should be 80 to 102 degrees).")
                        if not primary_state["leaning_ok"]:
                            feedback_details.append(f"Align forearm horizontally.")
                        if not primary_state["motion_ok"]:
                            feedback_details.append("Hold completely still.")
                        
                        status_msg.error(f"❌ Position Error: {' | '.join(feedback_details)}")
                else:
                    status_text = "ADJUST CAMERA"
                    color = (0, 165, 255)
                    status_msg.warning("⚠️ No forearm clearly visible. Align in camera.")

                if 'global_telemetry' in globals():
                    if not primary_state:
                        global_telemetry['status'] = "bad"
                        global_telemetry['message'] = "Align forearm clearly in camera view."
                        global_telemetry['accuracy'] = 45
                    elif final_ok:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Pose is correct. Forearm is positioned properly."
                        global_telemetry['accuracy'] = 95
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = 45
                        global_telemetry['message'] = f"Warning: {' | '.join(feedback_details)}"

                if primary_state:
                    draw_forearm(display_frame, primary_state, color)

                cv2.putText(display_frame, "TARGET: BACK ARM TRAUMA POSITION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            else:
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "bad"
                    global_telemetry['message'] = "Warning: Pose lost. Align your forearm inside the camera window bounds..."
                    global_telemetry['accuracy'] = 0
                status_msg.warning("⚠️ Pose lost. Align your forearm inside the camera window bounds...")
                cv2.putText(display_frame, "ADJUST CAMERA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            rgb_out = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_out, channels="RGB")

finally:
    pass
