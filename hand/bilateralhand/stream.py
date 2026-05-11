### Basic right and wrong code of bilateral hand with ML feedback and instructions
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
# # st.set_page_config(page_title="Bilateral PA Guide", layout="wide")
# st.title("🖐️ Bilateral PA Hand Assistant")

# # ==========================================
# # LOAD MODELS & ASSETS
# # ==========================================
# @st.cache_resource
# def load_bilateral_assets():
#     model = joblib.load("bilateral_pa_model.pkl")
#     label_encoder = joblib.load("bilateral_label_encoder.pkl")
    
#     MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=2,
#         min_hand_detection_confidence=0.4,
#         min_hand_presence_confidence=0.4,
#         min_tracking_confidence=0.4
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, label_encoder, detector = load_bilateral_assets()

# HAND_CONNECTIONS = [
#     (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
#     (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
#     (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # SIDEBAR SETTINGS
# # ==========================================
# st.sidebar.header("Camera")
# camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
# run_app = st.sidebar.checkbox("Start Assistant", value=True)

# # ==========================================
# # UI LAYOUT
# # ==========================================
# col_vid, col_inst = st.columns([2, 1])
# with col_vid:
#     frame_placeholder = st.empty()
# with col_inst:
#     st.subheader("🛠️ Guidance Commands")
#     cmd_box = st.empty()
#     tip_box = st.empty()

# # ==========================================
# # STABILITY STATE
# # ==========================================
# if 'state' not in st.session_state:
#     st.session_state.state = {"points": [], "label": "No Hand", "timer": 0}

# MAX_PERSISTENCE = 8

# # ==========================================
# # MAIN LOOP
# # ==========================================
# cap = cv2.VideoCapture(camera_index)

# while run_app:
#     ret, frame = cap.read()
#     if not ret: break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#     result = detector.detect(mp_image)

#     current_label = "No Hand"
#     current_points = []

#     if result.hand_landmarks and len(result.hand_landmarks) == 2:
#         st.session_state.state["timer"] = MAX_PERSISTENCE
        
#         # Sort hands by X to match training data
#         sorted_hands = sorted(result.hand_landmarks, key=lambda x: x[0].x)
        
#         row = []
#         for hand in sorted_hands:
#             pts = []
#             for lm in hand:
#                 row.extend([lm.x, lm.y, lm.z])
#                 pts.append((int(lm.x * w), int(lm.y * h)))
#             current_points.append(pts)

#         # Predict
#         X_in = pd.DataFrame([row])
#         pred_idx = model.predict(X_in)[0]
#         current_label = label_encoder.inverse_transform([pred_idx])[0]
        
#         st.session_state.state["label"] = current_label
#         st.session_state.state["points"] = current_points
#     else:
#         if st.session_state.state["timer"] > 0:
#             st.session_state.state["timer"] -= 1
#         else:
#             st.session_state.state["label"] = "No Hand"
#             st.session_state.state["points"] = []

#     # --- VISUAL FEEDBACK LOGIC ---
#     overlay_msg = st.session_state.state["label"]
    
#     if overlay_msg == "Bilateral PA":
#         color = (0, 255, 0) # Green
#         cmd_box.success("✅ Perfect Alignment!")
#         tip_box.info("Both hands are parallel and flat. Hold this position.")
    
#     elif overlay_msg == "Wrong":
#         color = (0, 0, 255) # Red
#         cmd_box.error("❌ Posture Incorrect")
#         tip_box.warning("""
#         **Commands to fix:**
#         1. Place **BOTH** hands flat on the table.
#         2. Keep palms facing **DOWN**.
#         3. Align your middle fingers so they are **parallel**.
#         4. Spread your fingers slightly to avoid overlapping.
#         """)
        
#         # DRAW GHOST SKELETON (Guidance Lines)
#         # This draws a "Reference" line to show where the hands should be
#         cv2.putText(frame, "FOLLOW GHOST GUIDE", (w//4, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
#     else:
#         color = (150, 150, 150)
#         cmd_box.info("Place both hands in view...")

#     # DRAW ACTUAL SKELETON
#     if st.session_state.state["points"]:
#         for hand_pts in st.session_state.state["points"]:
#             for conn in HAND_CONNECTIONS:
#                 cv2.line(frame, hand_pts[conn[0]], hand_pts[conn[1]], color, 3)
#             for pt in hand_pts:
#                 cv2.circle(frame, pt, 4, (255, 255, 255), -1)

#     cv2.putText(frame, overlay_msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#     frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
#     time.sleep(0.01)

# cap.release()


### This below code is the instructions based hand bilateral UI
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
# 1. HIGH-SPEED DUAL STABILIZER
# ==========================================
class BilateralStabilizer:
    def __init__(self, alpha=0.5): # Increased alpha for faster response
        self.alpha = alpha 
        self.prev_hands = None

    def stabilize(self, new_hands_list):
        if self.prev_hands is None or len(new_hands_list) != len(self.prev_hands):
            self.prev_hands = new_hands_list
            return new_hands_list
        
        stable_hands = []
        for i in range(len(new_hands_list)):
            stable_pts = self.prev_hands[i] * (1 - self.alpha) + new_hands_list[i] * self.alpha
            stable_hands.append(stable_pts)
        self.prev_hands = stable_hands
        return stable_hands

# ==========================================
# ASSETS & CONFIG
# ==========================================
@st.cache_resource
def load_bilateral_assets():
    model = joblib.load("bilateralhand/bilateral_pa_model.pkl")
    label_encoder = joblib.load("bilateralhand/bilateral_label_encoder.pkl")
    MODEL_PATH = "D:\\physio\\hand\\obliquehand\\hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_bilateral_assets()
stabilizer = BilateralStabilizer()

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP
# ==========================================
st.title("🖐️ Bilateral PA Hand Assistant")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Diagnostic Scan", value=True)
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
        status_color = (100, 100, 100)

        # UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            # Sort hands Left-to-Right for ML consistency
            sorted_hands = sorted(result.hand_landmarks, key=lambda x: x[0].x)
            raw_hands = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in sorted_hands]
            stable_hands = stabilizer.stabilize(raw_hands)
            
            # --- 1. PHYSICAL VALIDATION (Finger Extension) ---
            # Check if tips are above knuckles for all major fingers (8, 12, 16, 20)
            hands_opened = True
            for hand in stable_hands:
                tips = [hand[i][1] for i in [8, 12, 16, 20]]
                knuckles = [hand[i-2][1] for i in [8, 12, 16, 20]]
                if any(t > k for t, k in zip(tips, knuckles)):
                    hands_opened = False
            
            # --- 2. ML PREDICTION ---
            combined_row = np.concatenate([hand.flatten() for hand in stable_hands])
            prediction = model.predict(pd.DataFrame([combined_row]))[0]
            label = label_encoder.inverse_transform([prediction])[0]

            # --- DYNAMIC INSTRUCTIONS ---
            # Instruction: Opening Hands
            if hands_opened:
                checklist.append("[V] Hands fully opened")
            else:
                checklist.append("[X] Open both hands fully (Straighten fingers)")

            # Instruction: ML Match
            if label == "Bilateral PA" and hands_opened:
                checklist.append(f"[V] ML Match: {label}")
                is_ready = True
                status_color = (0, 255, 0) # Green Success
            else:
                checklist.append(f"[X] Posture: Incorrect")
                status_color = (0, 0, 255) # Red Error

            # Draw skeletons
            for hand_data in stable_hands:
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in hand_data]
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for pt in pixel_pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)

        else:
            checklist = ["[X] PLACE BOTH HANDS FULLY OPENED"] * 2

        # --- HIGHLIGHTED SUCCESS ---
        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 300, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Diagnostics
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 130 + (i*45)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()