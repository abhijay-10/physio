import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import math

# -------- ASSETS & CONFIG --------
model = joblib.load("pa_hand/pa_model.pkl")
MODEL_PATH = "d:\\physio\\hand\\pa_hand\\hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(7,8),
    (5,9),(9,10),(11,12), (9,13),(13,14),(15,16),
    (13,17),(17,18),(19,20), (0,17)
]

def draw_skeleton_placeholders(frame, h, w):
    overlay = frame.copy()
    start_x = 40
    line_width = int(w * 0.75)
    for i in range(5):
        y_pos = h - 165 + (i * 32)
        cv2.line(overlay, (start_x, y_pos), (start_x + line_width, y_pos), (80, 80, 80), 6)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def get_diagnostics(landmarks):
    p5, p17 = landmarks[5], landmarks[17]
    current_angle = math.degrees(math.atan2(p17.y - p5.y, p17.x - p5.x))
    rotation_needed = -int(current_angle) 
    return {
        "angle": abs(int(current_angle)),
        "rotation_needed": rotation_needed,
        "wrist_x": landmarks[0].x,
        "extended": landmarks[8].y < landmarks[5].y,
        "spread": abs(landmarks[8].x - landmarks[20].x)
    }

# -------- MEDIAPIPE SETUP --------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# -------- MAIN UI --------
st.title("🩺 Professional PA Hand Radiography Assistant")
camera_index = st.selectbox("Select Camera", [0,1,2,3], index=1)
run = st.checkbox("Start Diagnostic Scan")

frame_placeholder = st.empty()
timestamp = 0

if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        instructions = []
        is_fully_correct = False
        status_color = (150, 150, 150)

        # Draw UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                m = get_diagnostics(landmarks)
                
                # 1. Centering
                instructions.append("✅ Hand in frame" if 0.2 < m['wrist_x'] < 0.8 else "❌ Move hand to center")
                
                # 2. Rotation
                if m['angle'] < 10:
                    instructions.append(f"✅ Hand Flat (Tilt: {m['angle']}°)")
                else:
                    instructions.append(f"❌ Rotate hand {m['rotation_needed']}° to flat")
                
                # 3. Extension
                instructions.append("✅ Fingers extended" if m['extended'] else "❌ Straighten fingers")
                
                # 4. Spread
                instructions.append("✅ Digits spread" if m['spread'] > 0.18 else "❌ Spread fingers wider")
                
                # 5. ML Decision
                data_row = [landmarks[0].z, landmarks[8].z, landmarks[12].z, landmarks[16].z, landmarks[20].z, abs(landmarks[2].x - landmarks[17].x), landmarks[0].x]
                
                # FINAL VALIDATION CHECK
                if model.predict([data_row])[0] == 1 and m['angle'] < 10:
                    is_fully_correct = True
                    status_color = (0, 255, 0)
                    instructions.append("✅ READY FOR CAPTURE")
                else:
                    is_fully_correct = False
                    status_color = (0, 0, 255)
                    instructions.append("❌ POSTURE INCOMPLETE")

                # Draw Hand Skeleton
                for c in HAND_CONNECTIONS:
                    x1, y1 = int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h)
                    x2, y2 = int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h)
                    cv2.line(frame, (x1,y1), (x2,y2), status_color, 3)

        else:
            draw_skeleton_placeholders(frame, h, w)
            cv2.putText(frame, "SEARCHING FOR SENSOR...", (40, h-190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # -------- SUCCESS OVERLAY --------
        if is_fully_correct:
            # Large green banner for "Great" status
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 220, 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            # Add a subtle glow/border around the frame
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

        # Render Diagnostics Checklist
        for i, msg in enumerate(instructions):
            color = (0, 255, 0) if "✅" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 160 + (i*32)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_placeholder.image(frame, channels="BGR")
    cap.release()


# import streamlit as st
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import joblib
# import math
# import numpy as np

# # -------- ASSETS & CONFIG --------
# model = joblib.load("pa_model.pkl")
# MODEL_PATH = "d:\\physio\\pa_hand\\hand_landmarker.task"

# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(7,8),
#     (5,9),(9,10),(11,12), (9,13),(13,14),(15,16),
#     (13,17),(17,18),(19,20), (0,17)
# ]

# def check_all_landmarks_in_frame(landmarks):
#     """Ensures every single part of the hand is visible on screen."""
#     for lm in landmarks:
#         if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
#             return False
#     return True

# def get_diagnostics(landmarks):
#     p5, p17 = landmarks[5], landmarks[17]
#     current_angle = math.degrees(math.atan2(p17.y - p5.y, p17.x - p5.x))
#     rotation_needed = -int(current_angle) 
    
#     # Check if hand is too close or far (Z-axis)
#     avg_z = np.mean([lm.z for lm in landmarks])
    
#     return {
#         "angle": abs(int(current_angle)),
#         "rotation_needed": rotation_needed,
#         "wrist_x": landmarks[0].x,
#         "extended": landmarks[8].y < landmarks[5].y,
#         "spread": abs(landmarks[8].x - landmarks[20].x),
#         "in_frame": check_all_landmarks_in_frame(landmarks),
#         "depth": avg_z
#     }

# # -------- MEDIAPIPE SETUP --------
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=1
# )
# detector = vision.HandLandmarker.create_from_options(options)

# # -------- MAIN UI --------
# st.title("🩺 Professional PA Hand Radiography Assistant")
# camera_index = st.selectbox("Select Camera", [0,1,2,3], index=1)
# run = st.checkbox("Start Diagnostic Scan")

# frame_placeholder = st.empty()
# timestamp = 0

# if run:
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result = detector.detect_for_video(mp_image, timestamp)
#         timestamp += 1

#         instructions = []
#         is_fully_correct = False
#         status_color = (150, 150, 150)

#         # Draw UI Overlay Box
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
#         cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

#         if result.hand_landmarks:
#             for landmarks in result.hand_landmarks:
#                 m = get_diagnostics(landmarks)
                
#                 # ML Prediction with Reshape
#                 data_row = np.array([
#                     landmarks[0].z, landmarks[8].z, landmarks[12].z, 
#                     landmarks[16].z, landmarks[20].z, 
#                     abs(landmarks[2].x - landmarks[17].x), 
#                     landmarks[0].x
#                 ]).reshape(1, -1)
                
#                 ml_ready = model.predict(data_row)[0] == 1

#                 # 1. STRICT FRAME CHECK (Fixes your issue)
#                 instructions.append("[V] Hand fully in frame" if m['in_frame'] else "[X] Move hand fully into frame")
                
#                 # 2. ROTATION
#                 if m['angle'] < 10:
#                     instructions.append(f"[V] Hand Flat (Tilt: {m['angle']}deg)")
#                 else:
#                     instructions.append(f"[X] Rotate hand {m['rotation_needed']}deg to flat")
                
#                 # 3. EXTENSION
#                 instructions.append("[V] Fingers extended" if m['extended'] else "[X] Straighten fingers")
                
#                 # 4. SPREAD
#                 instructions.append("[V] Digits spread" if m['spread'] > 0.15 else "[X] Spread fingers wider")
                
#                 # 5. FINAL STATUS (Must meet ALL conditions)
#                 if ml_ready and m['in_frame'] and m['angle'] < 10:
#                     is_fully_correct = True
#                     status_color = (0, 255, 0)
#                     instructions.append("[V] STATUS: READY")
#                 else:
#                     is_fully_correct = False
#                     status_color = (0, 0, 255)
#                     instructions.append("[X] STATUS: INCOMPLETE")

#                 # Draw Skeleton
#                 for c in HAND_CONNECTIONS:
#                     x1, y1 = int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h)
#                     x2, y2 = int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h)
#                     cv2.line(frame, (x1,y1), (x2,y2), status_color, 3)

#         else:
#             instructions = ["[X] NO HAND DETECTED"] * 5

#         # -------- SUCCESS OVERLAY --------
#         if is_fully_correct:
#             cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 220, 100), 
#                         cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
#             cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

#         # Render Instructions
#         for i, msg in enumerate(instructions):
#             color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
#             cv2.putText(frame, msg, (45, h - 160 + (i*32)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         frame_placeholder.image(frame, channels="BGR")
#     cap.release()