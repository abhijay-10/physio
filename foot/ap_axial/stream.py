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
# 1. FOOT STABILIZER (LERP for Smooth Lines)
# ==========================================
class FootStabilizer:
    def __init__(self, alpha=0.45):
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
# ASSETS & CONFIG
# ==========================================
st.set_page_config(page_title="Physio AI - Foot", layout="wide")
st.title("🦶 Foot Radiography Assistant")

@st.cache_resource
def load_foot_assets():
    model = joblib.load("foot_model.pkl")
    label_encoder = joblib.load("foot_label_encoder.pkl")
    MODEL_PATH = "D:\\physio\\pose_landmarker_full.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_foot_assets()
stabilizer = FootStabilizer()

# Define the Foot Skeleton (Right Foot Focus: Ankle 28, Heel 30, Toe 32)
# We draw a triangle to represent the base of the foot
FOOT_SKELETON = [(28, 30), (30, 32), (32, 28)]

# ==========================================
# SIDEBAR - CAMERA SETTINGS
# ==========================================
st.sidebar.header("📷 Hardware")
camera_index = st.sidebar.selectbox("Select External Camera", options=[0, 1, 2, 3], index=1)
run_app = st.sidebar.checkbox("Start Diagnostic Scan", value=True)

frame_placeholder = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
if run_app:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        is_ready = False
        status_color = (120, 120, 120) # Gray

        # HUD Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            stable_pts = stabilizer.stabilize(raw_pts)
            pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

            # --- ML PREDICTION ---
            df = pd.DataFrame([stable_pts.flatten()])
            pred = model.predict(df)[0]
            label = label_encoder.inverse_transform([pred])[0]

            # --- DIAGNOSTIC MATH ---
            p_heel, p_toe = stable_pts[30], stable_pts[32]
            angle = abs(math.degrees(math.atan2(p_toe[1] - p_heel[1], p_toe[0] - p_heel[0])))

            # --- CHECKLIST ---
            # Angle Guidance
            if 80 < angle < 105:
                checklist.append(f"[V] AP Axial Alignment ({int(angle)}°)")
                angle_ok = True
            else:
                checklist.append(f"[X] Adjust Heel/Toe (Current: {int(angle)}°)")
                angle_ok = False

            # ML Result
            if label == "AP_Axial_Foot" and angle_ok:
                checklist.append(f"[V] ML Verified: {label}")
                is_ready = True
                status_color = (0, 255, 0) # Clinical Green
            else:
                checklist.append(f"[X] Posture Incorrect")
                status_color = (0, 0, 255) # Clinical Red

            # --- DRAW FOOT SKELETON ---
            # Drawing the connecting lines for the foot base
            for conn in FOOT_SKELETON:
                cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
            
            # Drawing the joint points
            for idx in [28, 30, 32]:
                cv2.circle(frame, pixel_pts[idx], 8, (255, 255, 255), -1) # White Joints
        else:
            checklist = ["[X] SEARCHING FOR FOOT LANDMARKS..."] * 2

        # --- SUCCESS FEEDBACK ---
        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT FOOT POSTURE", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Instructions
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 120 + (i*50)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()