# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # PAGE CONFIG
# # ==========================================
# # st.set_page_config(
# #     page_title="Hand Posture Detection",
# #     layout="wide"
# # )

# st.title("🖐️ Hand Posture Detection")

# # ==========================================
# # LOAD TRAINED MODEL
# # ==========================================
# model = joblib.load("obliquehand/hand_model.pkl")
# label_encoder = joblib.load("obliquehand/label_encoder.pkl")

# # ==========================================
# # LOAD MEDIAPIPE MODEL
# # ==========================================
# MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"

# base_options = python.BaseOptions(
#     model_asset_path=MODEL_PATH
# )

# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     num_hands=1,
#     min_hand_detection_confidence=0.5,
#     min_hand_presence_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# detector = vision.HandLandmarker.create_from_options(
#     options
# )

# # ==========================================
# # HAND CONNECTIONS
# # ==========================================
# HAND_CONNECTIONS = [

#     (0,1), (1,2), (2,3), (3,4),

#     (0,5), (5,6), (6,7), (7,8),

#     (0,9), (9,10), (10,11), (11,12),

#     (0,13), (13,14), (14,15), (15,16),

#     (0,17), (17,18), (18,19), (19,20),

#     (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # CAMERA
# # ==========================================
# camera_index = 1

# cap = cv2.VideoCapture(camera_index)

# if not cap.isOpened():

#     st.error("❌ Camera not opening")
#     st.stop()

# frame_placeholder = st.empty()

# # ==========================================
# # MAIN LOOP
# # ==========================================
# while True:

#     ret, frame = cap.read()

#     if not ret:
#         continue

#     frame = cv2.flip(frame, 1)

#     rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     mp_image = mp.Image(
#         image_format=mp.ImageFormat.SRGB,
#         data=rgb
#     )

#     result = detector.detect(mp_image)

#     prediction_text = "No Hand"

#     # ======================================
#     # DETECT HAND
#     # ======================================
#     if result.hand_landmarks:

#         h, w, _ = frame.shape

#         for hand_landmarks in result.hand_landmarks:

#             row = []
#             points = []

#             # --------------------------------
#             # LANDMARKS
#             # --------------------------------
#             for lm in hand_landmarks:

#                 row.extend([
#                     lm.x,
#                     lm.y,
#                     lm.z
#                 ])

#                 cx = int(lm.x * w)
#                 cy = int(lm.y * h)

#                 points.append((cx, cy))

#             # --------------------------------
#             # PREDICTION
#             # --------------------------------
#             X = pd.DataFrame([row])

#             prediction = model.predict(X)[0]

#             label = label_encoder.inverse_transform(
#                 [prediction]
#             )[0]

#             prediction_text = label

#             # --------------------------------
#             # COLORS
#             # --------------------------------
#             if label == "Left":

#                 color = (0,255,0)

#                 message = "✅ Correct LEFT Posture"

#             elif label == "Right":

#                 color = (255,0,0)

#                 message = "✅ Correct RIGHT Posture"

#             else:

#                 color = (0,0,255)

#                 message = "❌ Wrong Posture"

#             # --------------------------------
#             # DRAW SKELETON
#             # --------------------------------
#             for connection in HAND_CONNECTIONS:

#                 start_idx, end_idx = connection

#                 x1, y1 = points[start_idx]
#                 x2, y2 = points[end_idx]

#                 cv2.line(
#                     frame,
#                     (x1, y1),
#                     (x2, y2),
#                     color,
#                     3
#                 )

#             # --------------------------------
#             # DRAW LANDMARKS
#             # --------------------------------
#             for point in points:

#                 cv2.circle(
#                     frame,
#                     point,
#                     6,
#                     color,
#                     -1
#                 )

#             # --------------------------------
#             # SHOW TEXT
#             # --------------------------------
#             cv2.putText(
#                 frame,
#                 message,
#                 (20,50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 color,
#                 3
#             )

#     # ======================================
#     # DISPLAY FRAME
#     # ======================================
#     frame_placeholder.image(
#         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#         channels="RGB"
#     )

# # ==========================================
# # RELEASE
# # ==========================================
# cap.release()

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
# 1. STABLE INTERPOLATION CLASS
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
    model = joblib.load("obliquehand/hand_model.pkl")
    label_encoder = joblib.load("obliquehand/label_encoder.pkl")
    MODEL_PATH = "D:\\physio\\hand\\obliquehand\\hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.8
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
# SIDEBAR & UI
# ==========================================
st.sidebar.header("🎥 Hardware Setup")
cam_choice = st.sidebar.selectbox("Active Device Index", [0, 1, 2, 3], index=1)
run = st.sidebar.checkbox("Run Diagnostic Engine", value=True)

st.title("🩺 Advanced Oblique Hand Suite")
st.caption("Standardized 720p Analysis | Precision Diagnostic Checklist")

frame_placeholder = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
if run:
    cap = cv2.VideoCapture(cam_choice, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error(f"❌ Camera {cam_choice} offline.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret: break

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
            status_color = (80, 80, 80)

            # --- UI OVERLAY ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-180), (w, h), (12, 12, 12), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    stable_pts = stabilizer.stabilize(raw_pts)
                    pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                    # --- CORE PARAMETERS ---
                    # 1. Oblique Angle (45° Tilt)
                    p5, p17 = stable_pts[5], stable_pts[17]
                    angle = abs(math.degrees(math.atan2(p17[1] - p5[1], p17[0] - p5[0])))

                    # 2. Wrist Alignment (Horizontal Centering)
                    wrist_straight = abs(stable_pts[0][0] - stable_pts[9][0]) < 0.12
                    
                    # 3. Detection Depth Check (Z-axis)
                    avg_z = np.mean([p[2] for p in stable_pts])
                    depth_ok = -0.18 < avg_z < -0.02 

                    # ML Classification
                    if 25 < angle < 65:
                        df = pd.DataFrame([stable_pts.flatten()])
                        label = label_encoder.inverse_transform(model.predict(df))[0]
                    else:
                        label = "Unknown"

                    # --- FOCUSED CHECKLIST ---
                    # Parameter 1: Wrist
                    checklist.append("[V] Wrist Straight" if wrist_straight else "[X] Align Wrist")
                    
                    # Parameter 2: 45 Degree Goal
                    if 35 < angle < 55:
                        checklist.append(f"[V] Aim 45° Locked ({int(angle)}°)")
                    else:
                        checklist.append(f"[X] Aim 45° (Curr: {int(angle)}°)")
                    
                    # Parameter 3: Side Label
                    checklist.append(f"[V] Side: {label}" if label in ["Left", "Right"] else "[X] Detection Pending")

                    # FINAL SYNC
                    if (label in ["Left", "Right"] and 35 < angle < 55 and wrist_straight and depth_ok):
                        is_ready = True
                        status_color = (0, 255, 0)
                    else:
                        status_color = (0, 0, 255)

                    # Draw Skeleton
                    for conn in HAND_CONNECTIONS:
                        cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                    for pt in pixel_pts:
                        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            else:
                for i in range(3):
                    y_pos = h - 140 + (i * 35)
                    cv2.line(frame, (45, y_pos), (int(w * 0.5), y_pos), (60, 60, 60), 4)
                checklist = ["[X] SCANNING..."] * 3

            if is_ready:
                cv2.putText(frame, "GREAT! CORRECT OBLIQUE", (w//2 - 280, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
                cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

            # Render Focused Instructions
            for i, msg in enumerate(checklist):
                color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
                cv2.putText(frame, msg, (45, h - 130 + (i*35)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()