# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import math
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # 1. TORSO STABILIZER
# # ==========================================
# class TorsoStabilizer:
#     def __init__(self, alpha=0.5):
#         self.alpha = alpha 
#         self.prev_pts = None

#     def stabilize(self, new_pts):
#         if self.prev_pts is None:
#             self.prev_pts = new_pts
#             return new_pts
#         stable = self.prev_pts * (1 - self.alpha) + new_pts * self.alpha
#         self.prev_pts = stable
#         return stable

# # ==========================================
# # 2. ASSETS & CONFIG
# # ==========================================
# st.set_page_config(page_title="Physio AI - Lordotic Chest", layout="wide")
# st.title("🫁 Lordotic Chest Assistant")

# @st.cache_resource
# def load_assets():
#     # Load your trained Lordotic model and encoder
#     model = joblib.load("lordotic_model.pkl")
#     label_encoder = joblib.load("lordotic_label_encoder.pkl")
    
#     MODEL_PATH = "D:\physio\pose_landmarker_full.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.VIDEO,
#         num_poses=1,
#         min_pose_detection_confidence=0.5
#     )
#     detector = vision.PoseLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, encoder, detector = load_assets()
# stabilizer = TorsoStabilizer()

# # Define the Torso connections (Shoulders 11,12 & Hips 23,24)
# TORSO_SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24)]

# # ==========================================
# # 3. SIDEBAR HARDWARE SETTINGS
# # ==========================================
# st.sidebar.header("📷 Camera Settings")
# camera_index = st.sidebar.selectbox("Select DroidCam/Camera", options=[0, 1, 2, 3], index=1)
# run_app = st.sidebar.checkbox("Start Diagnostic Scan", value=True)

# frame_placeholder = st.empty()

# # ==========================================
# # 4. MAIN LOOP
# # ==========================================
# if run_app:
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
#         ms_timestamp = int(time.time() * 1000)
#         result = detector.detect_for_video(mp_image, ms_timestamp)

#         checklist = []
#         is_ready = False
#         status_color = (150, 150, 150) # Gray

#         # HUD Overlay Box
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
#         cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

#         if result.pose_landmarks:
#             landmarks = result.pose_landmarks[0]
#             raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
#             stable_pts = stabilizer.stabilize(raw_pts)
#             pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

#             # --- ML PREDICTION ---
#             # Data must be flattened to match your training set (99 features)
#             df = pd.DataFrame([stable_pts.flatten()])
#             pred = model.predict(df)[0]
#             label = encoder.inverse_transform([pred])[0]

#             # --- DIAGNOSTIC CALCULATION ---
#             # Shoulder level check (Landmarks 11 and 12)
#             shoulder_diff = abs(stable_pts[11][1] - stable_pts[12][1])
#             shoulders_level = shoulder_diff < 0.04
            
#             # Check for the backward lean (Lordotic Position)
#             # Typically shoulders move higher relative to the neck or depth changes
#             lean_detected = stable_pts[11][2] < stable_pts[23][2] # Simple Z-depth check

#             # --- CHECKLIST LOGIC ---
#             if shoulders_level:
#                 checklist.append("[V] Shoulders Level")
#             else:
#                 checklist.append("[X] Level your shoulders")

#             if label == "Correct_Lordotic":
#                 checklist.append("[V] Lordotic Lean Detected")
#                 if shoulders_level:
#                     is_ready = True
#                     status_color = (0, 255, 0) # Success Green
#             else:
#                 checklist.append("[X] Lean backward against sensor")
#                 status_color = (0, 0, 255) # Error Red

#             # Draw Skeleton
#             for conn in TORSO_SKELETON:
#                 cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
#             for idx in [11, 12, 23, 24]:
#                 cv2.circle(frame, pixel_pts[idx], 8, (255, 255, 255), -1)

#         else:
#             checklist = ["[X] POSITION TORSO IN FRAME"] * 2

#         # --- FINAL FEEDBACK ---
#         if is_ready:
#             cv2.putText(frame, "GREAT! LORDOTIC POSITION READY", (w//2 - 350, 80), 
#                         cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
#             cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

#         # Render Checklist Text
#         for i, msg in enumerate(checklist):
#             color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
#             cv2.putText(frame, msg, (45, h - 130 + (i*50)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#         frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
#     cap.release()


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
st.set_page_config(page_title="Physio AI - Lordotic Chest", layout="wide")
st.title("🫁 Lordotic Chest Assistant")

@st.cache_resource
def load_assets():
    model = joblib.load("lordotic_model.pkl")
    label_encoder = joblib.load("lordotic_label_encoder.pkl")
    
    MODEL_PATH = "D:\physio\pose_landmarker_full.task"
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
camera_index = st.sidebar.selectbox("Select DroidCam/Camera", options=[0, 1, 2, 3], index=1)
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
        if not ret: break

        # --- CRITICAL FIX FOR MIRROR PREDICTION ---
        # 1. We keep the raw_frame UNFLIPPED for the AI Model
        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. We flip the display_frame for the User (Mirror effect)
        display_frame = cv2.flip(frame, 1)
        
        h, w, _ = display_frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        is_ready = False
        status_color = (150, 150, 150) 

        # HUD UI on the flipped display frame
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            stable_pts = stabilizer.stabilize(raw_pts)
            
            # --- ML PREDICTION (Uses Unflipped Data) ---
            df = pd.DataFrame([stable_pts.flatten()])
            pred = model.predict(df)[0]
            label = encoder.inverse_transform([pred])[0]

            # --- DISPLAY DRAWING (Must be flipped to match mirror) ---
            # We subtract the X coordinate from 1 to mirror the skeleton lines
            pixel_pts = [(int((1 - p[0]) * w), int(p[1] * h)) for p in stable_pts]

            # --- CALCULATIONS ---
            shoulder_diff = abs(stable_pts[11][1] - stable_pts[12][1])
            shoulders_level = shoulder_diff < 0.04
            
            if shoulders_level:
                checklist.append("[V] Shoulders Level")
            else:
                checklist.append("[X] Level your shoulders")

            if label == "Correct_Lordotic":
                checklist.append("[V] Lordotic Position Detected")
                if shoulders_level:
                    is_ready = True
                    status_color = (0, 255, 0)
            else:
                checklist.append("[X] Incorrect Posture Detected")
                status_color = (0, 0, 255)

            # Draw Skeleton on Mirror View
            for conn in TORSO_SKELETON:
                cv2.line(display_frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
            for idx in [11, 12, 23, 24]:
                cv2.circle(display_frame, pixel_pts[idx], 8, (255, 255, 255), -1)

        else:
            checklist = ["[X] POSITION TORSO IN FRAME"] * 2

        if is_ready:
            cv2.putText(display_frame, "GREAT! LORDOTIC POSITION READY", (w//2 - 350, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(display_frame, (0,0), (w,h), (0, 255, 0), 12)

        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(display_frame, msg, (45, h - 130 + (i*50)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()

