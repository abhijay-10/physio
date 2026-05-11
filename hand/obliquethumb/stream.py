import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. STABLE INTERPOLATION (Zero Flickering)
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
    model = joblib.load("obliquethumb/oblique_thumb_model.pkl")
    label_encoder = joblib.load("obliquethumb/oblique_label_encoder.pkl")
    MODEL_PATH = "d:\\physio\\hand\\obliquehand\\hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
stabilizer = HandStabilizer()

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("🩺 Precision Oblique Thumb Diagnostic")
camera_index = st.sidebar.selectbox("Camera Device", options=[0, 1, 2, 3], index=0)
run_app = st.sidebar.checkbox("Start Live Analysis", value=True)

frame_placeholder = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
if run_app:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        # Success trackers
        checks_passed = 0 
        is_fully_correct = False
        status_color = (100, 100, 100) 

        # Background Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                stable_pts = stabilizer.stabilize(raw_pts)
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # --- 1. ML PREDICTION ---
                df = pd.DataFrame([stable_pts.flatten()])
                prediction = model.predict(df)[0]
                label = label_encoder.inverse_transform([prediction])[0]

                # --- 2. ANGLE GUIDANCE ---
                p2, p4 = stable_pts[2], stable_pts[4]
                angle = abs(math.degrees(math.atan2(p4[1] - p2[1], p4[0] - p2[0])))
                
                # --- 3. REFINED CURL LOGIC ---
                # Measure distance from palm (0) to knuckles (5, 9, 13, 17)
                # If the hand is curled, the finger tips (8, 12, 16, 20) are near the palm
                curl_dist = np.mean([abs(stable_pts[i][1] - stable_pts[0][1]) for i in [8, 12, 16, 20]])
                is_curled = curl_dist < 0.28 

                # --- BUILD CHECKLIST ---
                # Check 1: Curl
                if is_curled:
                    checklist.append("[V] Curl your fingers")
                    checks_passed += 1
                else:
                    checklist.append("[X] Curl your fingers")

                # Check 2: Thumb Straightness
                if stable_pts[4][1] < stable_pts[2][1]:
                    checklist.append("[V] Straighten your thumb")
                    checks_passed += 1
                else:
                    checklist.append("[X] Straighten your thumb")

                # Check 3: 45 Degree Angle
                if 35 < angle < 55:
                    checklist.append(f"[V] 45 degree tilt achieved ({int(angle)} deg)")
                    checks_passed += 1
                else:
                    checklist.append(f"[X] 45 degree tilt slight forward ({int(angle)} deg)")

                # Check 4: ML Prediction Match
                if label == "Oblique Thumb":
                    checks_passed += 1

                # --- GLOBAL SYNC: TRIGGER SUCCESS ---
                # Only if all parameters (including ML) are satisfied
                if checks_passed == 4:
                    is_fully_correct = True
                    status_color = (0, 255, 0) # ALL GREEN
                else:
                    status_color = (0, 0, 255) # RED if any fail

                # Draw Visuals
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for pt in pixel_pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        else:
            checklist = ["[X] POSITION THUMB FOR SCAN"] * 3

        # --- HIGHLIGHTED SUCCESS FEEDBACK ---
        if is_fully_correct:
            # Large Success Banner
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 320, 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 3)
            # Glowing Full-Screen Border
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Instructions
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 135 + (i*40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()