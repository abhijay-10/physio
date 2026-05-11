

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
# 1. STABLE INTERPOLATION
# ==========================================
class HandStabilizer:
    def __init__(self, alpha=0.45):
        self.alpha = alpha 
        self.prev_points = None

    def stabilize(self, new_points):
        if self.prev_points is None:
            self.prev_points = new_points
            return new_points
        stable_points = self.prev_points * (1 - self.alpha) + new_points * self.alpha
        self.prev_points = stable_points
        return stable_points

# ==========================================
# ASSETS & CONFIG
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("pa3finger/pa_finger_model.pkl")
    label_encoder = joblib.load("pa3finger/pa_finger_label_encoder.pkl")
    MODEL_PATH = "D:\\physio\\hand\\obliquehand\\hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
stabilizer = HandStabilizer(alpha=0.4)

# Target connections: Index, Middle, Ring
PA_CONNECTIONS = [
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16)
]
ALLOWED_PTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

# ==========================================
# MAIN UI
# ==========================================
st.title("🖐️ 3-Finger PA Diagnostic Suite")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Live Analysis", value=True)
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        is_ready = False
        status_color = (150, 150, 150)

        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                stable_pts = stabilizer.stabilize(np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]))
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # --- MULTI-FINGER VALIDATION ---
                # Verify that Index(8), Middle(12), and Ring(16) are all extended
                fingers_extended = []
                for tip, knuckle in [(8, 5), (12, 9), (16, 13)]:
                    if stable_pts[tip][1] < stable_pts[knuckle][1]:
                        fingers_extended.append(True)
                
                digit_count = len(fingers_extended)
                
                # ML Classification
                df = pd.DataFrame([stable_pts.flatten()])
                label = label_encoder.inverse_transform(model.predict(df))[0]

                # --- CHECKLIST LOGIC ---
                # 1. Count Check
                if digit_count == 3:
                    checklist.append("[V] All 3 Fingers Extended")
                else:
                    checklist.append(f"[X] Open 3 Fingers (Detected: {digit_count})")

                # 2. Separation Check
                spread = abs(stable_pts[8][0] - stable_pts[16][0])
                checklist.append("[V] Fingers Separated" if spread > 0.18 else "[X] Separate fingers wider")

                # 3. Posture Verification
                if label == "PA Finger" and digit_count == 3:
                    checklist.append("[V] ML Posture Verified")
                    if spread > 0.18:
                        is_ready = True
                        status_color = (0, 255, 0)
                else:
                    checklist.append("[X] Seeking PA Posture...")
                    status_color = (0, 0, 255)

                # Draw Visuals
                for conn in PA_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for idx in ALLOWED_PTS:
                    cv2.circle(frame, pixel_pts[idx], 4, (255, 255, 255), -1)
        else:
            checklist = ["[X] POSITION 3 FINGERS FOR SCAN"] * 3

        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT PA POSTURE", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 130 + (i*40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()