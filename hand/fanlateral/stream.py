# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib
# import time

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # PAGE CONFIG
# # ==========================================
# # st.set_page_config(page_title="Fan Lateral Tester", layout="wide")
# st.title("🖐️ Fan Lateral Posture Testing")
# st.markdown("This app detects **Left Fan Lateral** and **Right Fan Lateral** postures.")

# # ==========================================
# # SIDEBAR - CAMERA & CONTROLS
# # ==========================================
# st.sidebar.header("Testing Settings")
# camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
# run_test = st.sidebar.checkbox("Start Camera", value=True)

# # ==========================================
# # LOAD ASSETS (Cached)
# # ==========================================
# @st.cache_resource
# def load_fan_model():
#     # Loading the model you just trained
#     model = joblib.load("fanhand_model.pkl")
#     label_encoder = joblib.load("fanlabel_encoder.pkl")
    
#     # MediaPipe Setup
#     MODEL_PATH = "d:\\physio\\obliquehand\\hand_landmarker.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=1,
#         min_hand_detection_confidence=0.4,
#         min_hand_presence_confidence=0.4,
#         min_tracking_confidence=0.4
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, label_encoder, detector = load_fan_model()

# # Standard Hand Skeleton
# HAND_CONNECTIONS = [
#     (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
#     (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
#     (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # PERSISTENCE STATE (No Flickering)
# # ==========================================
# if 'test_points' not in st.session_state:
#     st.session_state.test_points = None
#     st.session_state.test_msg = "No Hand Detected"
#     st.session_state.test_color = (128, 128, 128)
#     st.session_state.test_timer = 0

# MAX_HOLD = 10 # Number of frames to hold lines during detection drops

# # ==========================================
# # MAIN LOOP
# # ==========================================
# frame_placeholder = st.empty()
# cap = cv2.VideoCapture(camera_index)

# while run_test:
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Camera index not found. Try switching settings in the sidebar.")
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
#     result = detector.detect(mp_image)

#     if result.hand_landmarks:
#         st.session_state.test_timer = MAX_HOLD
        
#         for hand_landmarks in result.hand_landmarks:
#             row = []
#             points = []
#             for lm in hand_landmarks:
#                 row.extend([lm.x, lm.y, lm.z])
#                 points.append((int(lm.x * w), int(lm.y * h)))

#             # PREDICTION
#             X_input = pd.DataFrame([row])
#             prediction = model.predict(X_input)[0]
#             label = label_encoder.inverse_transform([prediction])[0]

#             # LABEL LOGIC
#             if "Left" in label:
#                 st.session_state.test_msg = "✅ SUCCESS: LEFT FAN LATERAL"
#                 st.session_state.test_color = (0, 255, 0) # Green
#             elif "Right" in label:
#                 st.session_state.test_msg = "✅ SUCCESS: RIGHT FAN LATERAL"
#                 st.session_state.test_color = (255, 140, 0) # Orange/Cyan
#             else:
#                 st.session_state.test_msg = "❌ WRONG POSTURE"
#                 st.session_state.test_color = (0, 0, 255) # Red
            
#             st.session_state.test_points = points

#     else:
#         # Persistence check
#         if st.session_state.test_timer > 0:
#             st.session_state.test_timer -= 1
#         else:
#             st.session_state.test_points = None
#             st.session_state.test_msg = "Ready to Test..."
#             st.session_state.test_color = (150, 150, 150)

#     # DRAWING (STABLE)
#     if st.session_state.test_points:
#         # Draw Skeleton
#         for conn in HAND_CONNECTIONS:
#             cv2.line(frame, st.session_state.test_points[conn[0]], 
#                      st.session_state.test_points[conn[1]], 
#                      st.session_state.test_color, 3)
#         # Draw Points
#         for pt in st.session_state.test_points:
#             cv2.circle(frame, pt, 5, (255, 255, 255), -1)

#     # UI OVERLAY
#     cv2.putText(frame, st.session_state.test_msg, (30, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, st.session_state.test_color, 3)

#     # DISPLAY
#     frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
#     # Tiny delay to prevent Streamlit from overloading
#     time.sleep(0.01)

# cap.release()


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
# 1. STABLE INTERPOLATION (Zero Flickering)
# ==========================================
class HandStabilizer:
    def __init__(self, alpha=0.45):
        self.alpha = alpha 
        self.prev_points = None

    def stabilize(self, new_points):
        if self.prev_points is None:
            return new_points
        if self.prev_points is None:
            self.prev_points = new_points
            return new_points
        stable_points = self.prev_points * (1 - self.alpha) + new_points * self.alpha
        self.prev_points = stable_points
        return stable_points

# ==========================================
# ASSETS & CONFIG
# ==========================================
st.title("🩺 Fan Lateral Diagnostic Assistant")

@st.cache_resource
def load_fan_assets():
    # Loading your specific Fan Lateral models
    model = joblib.load("fanlateral/fanhand_model.pkl")
    label_encoder = joblib.load("fanlateral/fanlabel_encoder.pkl")
    MODEL_PATH = "d:\\physio\\hand\\obliquehand\\hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_fan_assets()
stabilizer = HandStabilizer(alpha=0.45)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP SETUP
# ==========================================
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Live Analysis", value=True)
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Normalize View
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Fixed Monotonic Timestamp
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        is_ready = False
        status_color = (150, 150, 150) # Neutral Gray

        # UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 1. Smoothing
                raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                stable_pts = stabilizer.stabilize(raw_pts)
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # 2. ML Prediction
                df = pd.DataFrame([stable_pts.flatten()])
                pred = model.predict(df)[0]
                label = label_encoder.inverse_transform([pred])[0]

                # 3. DIAGNOSTICS (Fan Specific)
                # Wrist Verticality check (targeting ~90 deg)
                p0, p9 = stable_pts[0], stable_pts[9]
                angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))
                
                # Fan Separation check (Spread between Index and Pinky tips)
                spread = abs(stable_pts[8][0] - stable_pts[20][0])
                
                # --- CLINICAL CHECKLIST ---
                # 1. Centering
                checklist.append("[V] Hand Centered" if 0.2 < stable_pts[0][0] < 0.8 else "[X] Move hand to center")
                
                # 2. Verticality
                if 75 < angle < 105:
                    checklist.append(f"[V] Vertical Alignment ({int(angle)}deg)")
                else:
                    checklist.append(f"[X] Align Wrist Vertically ({int(angle)}deg)")
                
                # 3. Fan Spread (Critical for Fan Lateral)
                checklist.append("[V] Fingers Fanned (Separated)" if spread > 0.22 else "[X] Separate fingers like a fan")

                # 4. Posture Match
                if "Fan Lateral" in label:
                    checklist.append(f"[V] Detected: {label}")
                    if 75 < angle < 105 and spread > 0.22:
                        is_ready = True
                        status_color = (0, 255, 0)
                else:
                    checklist.append("[X] Adjusting to Fan Posture...")
                    status_color = (0, 0, 255)

                # Draw Visual Skeleton
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for pt in pixel_pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        else:
            checklist = ["[X] POSITION HAND FOR FAN SCAN"] * 3

        # --- SUCCESS FEEDBACK ---
        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT FAN LATERAL", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Diagnostics
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 160 + (i*40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()