import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# PAGE CONFIG
# ==========================================
# st.set_page_config(page_title="PA Finger Guide", layout="wide")
st.title("🖐️ PA Finger Posture Guide")
st.markdown("Follow the instructions below to achieve the perfect **PA Finger (Index, Middle, Ring)** posture.")

# ==========================================
# SIDEBAR - CAMERA SETTINGS
# ==========================================
st.sidebar.header("Settings")
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
run_app = st.sidebar.checkbox("Start Live Feed", value=True)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("pa_finger_model.pkl")
    label_encoder = joblib.load("pa_finger_label_encoder.pkl")
    
    MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()

# First 3 fingers connections (Index, Middle, Ring)
PA_CONNECTIONS = [
    (0,5), (5,6), (6,7), (7,8),       # Index
    (0,9), (9,10), (10,11), (11,12),    # Middle
    (0,13), (13,14), (14,15), (15,16)   # Ring
]
ALLOWED_PTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

# ==========================================
# STABILITY & INSTRUCTION STATE
# ==========================================
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = "Searching"
    st.session_state.points = None
    st.session_state.timer = 0

MAX_HOLD = 10

# ==========================================
# UI LAYOUT
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("📋 Status & Instructions")
    status_container = st.empty()
    instruction_container = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(camera_index)

while run_app:
    ret, frame = cap.read()
    if not ret:
        st.warning("Waiting for camera...")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        st.session_state.timer = MAX_HOLD
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))

            # Predict
            X_input = pd.DataFrame([row])
            pred_idx = model.predict(X_input)[0]
            label = label_encoder.inverse_transform([pred_idx])[0]
            
            st.session_state.last_pred = label
            st.session_state.points = points
    else:
        if st.session_state.timer > 0:
            st.session_state.timer -= 1
        else:
            st.session_state.last_pred = "No Hand"
            st.session_state.points = None

    # --- DRAWING & FEEDBACK LOGIC ---
    display_msg = "No Hand Detected"
    display_color = (150, 150, 150)
    
    if st.session_state.last_pred == "PA Finger":
        display_msg = "✅ PERFECT PA POSTURE"
        display_color = (0, 255, 0)
        status_container.success(display_msg)
        instruction_container.info("Great! Keep your fingers flat and separated as shown.")
    
    elif st.session_state.last_pred == "Wrong":
        display_msg = "❌ WRONG POSTURE"
        display_color = (0, 0, 255)
        status_container.error(display_msg)
        instruction_container.warning("""
        **How to fix:**
        1. Lay your Index, Middle, and Ring fingers **flat** on the surface.
        2. Ensure they are **straight** and not curled.
        3. Keep a small gap between each finger.
        4. Make sure your palm is facing **down**.
        """)
    else:
        status_container.info("Searching for hand...")

    # Draw the skeleton (Only the 3 fingers)
    if st.session_state.points:
        for conn in PA_CONNECTIONS:
            cv2.line(frame, st.session_state.points[conn[0]], 
                     st.session_state.points[conn[1]], display_color, 3)
        for i in ALLOWED_PTS:
            cv2.circle(frame, st.session_state.points[i], 5, (255, 255, 255), -1)

    cv2.putText(frame, display_msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 3)
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    time.sleep(0.01)

cap.release()