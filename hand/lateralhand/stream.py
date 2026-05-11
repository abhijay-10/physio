


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
# 1. LANDMARK SMOOTHING
# ==========================================
class LandmarkSmoother:
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.history = []

    def smooth(self, new_landmarks):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in new_landmarks])
        self.history.append(coords)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

# ==========================================
# ASSETS & CONFIG
# ==========================================
# st.set_page_config(page_title="Lateral Radiography UI", layout="wide")
st.title("✋ Clinical Lateral Hand Assistant")

@st.cache_resource
def load_assets():
    model = joblib.load("lateralhand/lateral_dual_hand_model.pkl")
    label_encoder = joblib.load("lateralhand/lateral_dual_label_encoder.pkl")
    MODEL_PATH = "d:\\physio\\hand\\obliquehand\\hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
smoother = LandmarkSmoother(window_size=8) # Smooths over 8 frames for stability

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP
# ==========================================
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
run_app = st.sidebar.checkbox("Start Scan", value=True)

frame_placeholder = st.empty()
cap = cv2.VideoCapture(camera_index)

while run_app:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    
    checklist = []
    success_count = 0
    header_text = "ALIGNING..."
    status_color = (0, 0, 255) # Start Red

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Smooth the jitter
            smoothed = smoother.smooth(hand_landmarks)
            pts = [(int(c[0] * w), int(c[1] * h)) for c in smoothed]
            
            # ML Data
            row = smoothed.flatten()
            pred = model.predict(pd.DataFrame([row]))[0]
            label = label_encoder.inverse_transform([pred])[0]

            # Angle Calc (Verticality)
            p0, p9 = smoothed[0], smoothed[9]
            angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))

            # --- 5 POINT DIAGNOSTIC CHECK ---
            # 1. Centering
            if 0.3 < smoothed[0][0] < 0.7:
                checklist.append("[V] Hand Centered")
                success_count += 1
            else:
                checklist.append("[X] Move hand to center")

            # 2. Angle
            if 78 < angle < 102:
                checklist.append(f"[V] Vertical Alignment ({int(angle)}deg)")
                success_count += 1
            else:
                checklist.append(f"[X] Adjust Hand Angle ({int(angle)}deg)")

            # 3. Fingertips
            if all(0.1 < c[1] < 0.9 for c in smoothed):
                checklist.append("[V] All digits in frame")
                success_count += 1
            else:
                checklist.append("[X] Fingers out of frame")

            # 4. ML Posture Identification
            if label == "Left" or label == "Right":
                checklist.append(f"[V] Detected: {label} Lateral")
                success_count += 1
            else:
                checklist.append("[X] Unknown Hand Posture")

            # 5. Stability/Spread
            if abs(smoothed[4][0] - smoothed[5][0]) > 0.02:
                checklist.append("[V] Neutral thumb position")
                success_count += 1
            else:
                checklist.append("[X] Move thumb away from palm")

            # --- CHECK FINAL SYNC ---
            if success_count == 5:
                status_color = (0, 255, 0)
                header_text = "GREAT! CORRECT POSTURE"
            else:
                status_color = (0, 0, 255)
                header_text = "POSTURE INCOMPLETE"

            # Draw Smoothed Skeleton
            for conn in HAND_CONNECTIONS:
                cv2.line(frame, pts[conn[0]], pts[conn[1]], status_color, 3)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
    else:
        checklist = ["[X] NO HAND DETECTED"] * 5

    # --- CLINICAL UI ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-225), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if success_count == 5:
        cv2.putText(frame, header_text, (w//2 - 250, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)
    else:
        cv2.putText(frame, f"STATUS: {header_text}", (20, h-195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    for i, msg in enumerate(checklist):
        color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
        cv2.putText(frame, msg, (35, h - 160 + (i*32)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()