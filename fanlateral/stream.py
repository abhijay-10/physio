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
# st.set_page_config(page_title="Fan Lateral Tester", layout="wide")
st.title("🖐️ Fan Lateral Posture Testing")
st.markdown("This app detects **Left Fan Lateral** and **Right Fan Lateral** postures.")

# ==========================================
# SIDEBAR - CAMERA & CONTROLS
# ==========================================
st.sidebar.header("Testing Settings")
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
run_test = st.sidebar.checkbox("Start Camera", value=True)

# ==========================================
# LOAD ASSETS (Cached)
# ==========================================
@st.cache_resource
def load_fan_model():
    # Loading the model you just trained
    model = joblib.load("fanlateral/fanhand_model.pkl")
    label_encoder = joblib.load("fanlateral/fanlabel_encoder.pkl")
    
    # MediaPipe Setup
    MODEL_PATH = "d:\\physio\\obliquehand\\hand_landmarker.task"
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

model, label_encoder, detector = load_fan_model()

# Standard Hand Skeleton
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# PERSISTENCE STATE (No Flickering)
# ==========================================
if 'test_points' not in st.session_state:
    st.session_state.test_points = None
    st.session_state.test_msg = "No Hand Detected"
    st.session_state.test_color = (128, 128, 128)
    st.session_state.test_timer = 0

MAX_HOLD = 10 # Number of frames to hold lines during detection drops

# ==========================================
# MAIN LOOP
# ==========================================
frame_placeholder = st.empty()
cap = cv2.VideoCapture(camera_index)

while run_test:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera index not found. Try switching settings in the sidebar.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        st.session_state.test_timer = MAX_HOLD
        
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))

            # PREDICTION
            X_input = pd.DataFrame([row])
            prediction = model.predict(X_input)[0]
            label = label_encoder.inverse_transform([prediction])[0]

            # LABEL LOGIC
            if "Left" in label:
                st.session_state.test_msg = "✅ SUCCESS: LEFT FAN LATERAL"
                st.session_state.test_color = (0, 255, 0) # Green
            elif "Right" in label:
                st.session_state.test_msg = "✅ SUCCESS: RIGHT FAN LATERAL"
                st.session_state.test_color = (255, 140, 0) # Orange/Cyan
            else:
                st.session_state.test_msg = "❌ WRONG POSTURE"
                st.session_state.test_color = (0, 0, 255) # Red
            
            st.session_state.test_points = points

    else:
        # Persistence check
        if st.session_state.test_timer > 0:
            st.session_state.test_timer -= 1
        else:
            st.session_state.test_points = None
            st.session_state.test_msg = "Ready to Test..."
            st.session_state.test_color = (150, 150, 150)

    # DRAWING (STABLE)
    if st.session_state.test_points:
        # Draw Skeleton
        for conn in HAND_CONNECTIONS:
            cv2.line(frame, st.session_state.test_points[conn[0]], 
                     st.session_state.test_points[conn[1]], 
                     st.session_state.test_color, 3)
        # Draw Points
        for pt in st.session_state.test_points:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)

    # UI OVERLAY
    cv2.putText(frame, st.session_state.test_msg, (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, st.session_state.test_color, 3)

    # DISPLAY
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    # Tiny delay to prevent Streamlit from overloading
    time.sleep(0.01)

cap.release()