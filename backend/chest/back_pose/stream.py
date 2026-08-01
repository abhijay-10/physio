import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. TORSO STABILIZER
# ==========================================
class TorsoStabilizer:
    def __init__(self, alpha=0.15): 
        self.alpha = alpha 
        self.prev_pts = None

    def stabilize(self, new_pts):
        if self.prev_pts is None:
            self.prev_pts = new_pts
            return new_pts
        stable = self.prev_pts * (1 - self.alpha) + new_pts * self.alpha
        self.prev_pts = stable
        return stable

# ==========================================
# 2. ASSETS & CONFIG
# ==========================================
st.set_page_config(page_title="Physio AI - PA Chest", layout="wide")
st.title("🫁 Chest Back-Pose (PA View) Assistant")

@st.cache_resource
def load_assets():
    # Load your trained Back-Pose model and encoder
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/back_pose/back_pose_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/back_pose/back_pose_label_encoder.pkl").replace("\\", "/"))
    
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5 
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, encoder, detector = load_assets()
stabilizer = TorsoStabilizer()

TORSO_SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24)]

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.header("📷 Camera Settings")
camera_index = st.sidebar.selectbox("Select DroidCam/Camera", options=[0, 2], format_func=lambda x: "Laptop Camera" if x==0 else "Droid Camera", index=0)
run_app = st.sidebar.checkbox("Start Diagnostic Scan", value=True)

frame_placeholder = st.empty()

# ==========================================
# 4. MAIN LOOP
# ==========================================
if run_app:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: 
            if 'active_stop_event' in globals() and active_stop_event.is_set(): break
            import time
            time.sleep(0.01)
            continue

        # --- MIRROR LOGIC ---
        display_frame = cv2.flip(frame, 1)              # User sees this
        h, w, _ = display_frame.shape

        # HUD Overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            stable_pts = stabilizer.stabilize(raw_pts)
            
            # --- ML PREDICTION (Unflipped) ---
            pred = model.predict([stable_pts.flatten()])[0]
            label = encoder.inverse_transform([pred])[0]

            # --- DISPLAY DRAWING (Mirrored) ---
            pixel_pts = [(int((1 - p[0]) * w), int(p[1] * h)) for p in stable_pts]

            # --- CLINICAL CHECKS ---
            shoulder_diff = abs(stable_pts[11][1] - stable_pts[12][1])
            shoulders_level = shoulder_diff < 0.04
            
            checklist = []
            if shoulders_level:
                checklist.append("[V] Shoulders Level")
            else:
                checklist.append("[X] Level your shoulders")

            shoulder_depth_diff = abs(stable_pts[11][2] - stable_pts[12][2])
            not_tilted = shoulder_depth_diff < 0.15  # Ensure person is facing flat away, not turned sideways

            if label == "Correct_Back_Pose" and not_tilted:
                checklist.append("[V] Back-Pose Detected")
                if shoulders_level:
                    is_ready = True
                    status_color = (0, 255, 0)
                else:
                    is_ready = False
                    status_color = (150, 150, 150)
            else:
                if not not_tilted and label == "Correct_Back_Pose":
                    checklist.append("[X] Stand flat (Do not tilt)")
                else:
                    checklist.append("[X] Turn back to camera")
                is_ready = False
                status_color = (0, 0, 255)
        else:
            pixel_pts = None
            checklist = ["[X] POSITION BACK TO CAMERA"] * 2
            is_ready = False
            status_color = (150, 150, 150)

        if pixel_pts:
            # Draw Skeleton
            for conn in TORSO_SKELETON:
                cv2.line(display_frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
            for idx in [11, 12, 23, 24]:
                cv2.circle(display_frame, pixel_pts[idx], 8, (255, 255, 255), -1)

        if is_ready:
            cv2.putText(display_frame, "READY: CAPTURE PA CHEST VIEW", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(display_frame, (0,0), (w,h), (0, 255, 0), 12)

        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(display_frame, msg, (45, h - 130 + (i*50)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # -- INJECT TELEMETRY FOR FRONTEND --
        if 'global_telemetry' in globals():
            local_status = locals().get('is_fully_correct', locals().get('is_ready', False))
            local_msgs = locals().get('instructions', locals().get('checklist', []))
            fail_msgs = [m for m in local_msgs if "[FAIL]" in m or "[X]" in m]
            if local_status:
                global_telemetry['message'] = "Perfect alignment. Keep holding."
                global_telemetry['accuracy'] = 95
                global_telemetry['status'] = "good"
            elif fail_msgs:
                global_telemetry['message'] = fail_msgs[0].replace("[FAIL] ", "Warning: ").replace("[X] ", "Warning: ")
                global_telemetry['accuracy'] = 45
                global_telemetry['status'] = "bad"
            else:
                global_telemetry['message'] = "Analyzing..."
                global_telemetry['accuracy'] = 10
                global_telemetry['status'] = "calibrating"
        import time
        time.sleep(0.01) # Yield GIL
        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()