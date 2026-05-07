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
# st.set_page_config(page_title="Oblique Thumb Detector", layout="wide")
st.title("👍 Oblique Thumb Posture Detection")

# ==========================================
# SIDEBAR - SETTINGS
# ==========================================
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.selectbox("Select Camera Index", options=[0, 1, 2], index=0)
run_app = st.sidebar.checkbox("Start Detection", value=True)

# ==========================================
# LOAD ASSETS (Cached for Speed)
# ==========================================
@st.cache_resource
def load_oblique_assets():
    # Make sure these filenames match your training script output
    model = joblib.load("obliquethumb/oblique_thumb_model.pkl")
    label_encoder = joblib.load("obliquethumb/oblique_label_encoder.pkl")
    
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

model, label_encoder, detector = load_oblique_assets()

# Hand Skeleton Connections
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]
THUMB_INDICES = [0, 1, 2, 3, 4]

# ==========================================
# STABILITY STATE (Zero Flickering)
# ==========================================
if 'obs_points' not in st.session_state:
    st.session_state.obs_points = None
    st.session_state.obs_msg = "No Hand Detected"
    st.session_state.obs_color = (128, 128, 128)
    st.session_state.obs_timer = 0

MAX_HOLD = 10 # Frames to hold the skeleton during a flicker

# ==========================================
# MAIN INTERFACE
# ==========================================
frame_placeholder = st.empty()
cap = cv2.VideoCapture(camera_index)

while run_app:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not found. Check sidebar settings.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        st.session_state.obs_timer = MAX_HOLD
        
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))

            # PREDICTION
            X_data = pd.DataFrame([row])
            pred = model.predict(X_data)[0]
            label = label_encoder.inverse_transform([pred])[0]

            # UPDATE VISUALS
            if label == "Oblique Thumb":
                st.session_state.obs_msg = "✅ Correct: OBLIQUE THUMB"
                st.session_state.obs_color = (0, 255, 0) # Green
            else:
                st.session_state.obs_msg = "❌ WRONG POSTURE"
                st.session_state.obs_color = (0, 0, 255) # Red
            
            st.session_state.obs_points = points
    else:
        # Persistence Countdown
        if st.session_state.obs_timer > 0:
            st.session_state.obs_timer -= 1
        else:
            st.session_state.obs_points = None
            st.session_state.obs_msg = "Ready to Detect..."
            st.session_state.obs_color = (150, 150, 150)

    # DRAWING (Using state memory to stop flickering)
    if st.session_state.obs_points:
        for conn in HAND_CONNECTIONS:
            # Highlight thumb in Green if Correct, otherwise keep theme color
            c = (0, 255, 0) if (conn[0] in THUMB_INDICES and conn[1] in THUMB_INDICES and st.session_state.obs_msg.startswith("✅")) else st.session_state.obs_color
            cv2.line(frame, st.session_state.obs_points[conn[0]], st.session_state.obs_points[conn[1]], c, 3)
            
        for pt in st.session_state.obs_points:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)

    # UI OVERLAY
    cv2.putText(frame, st.session_state.obs_msg, (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, st.session_state.obs_color, 3)

    # RENDER
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    time.sleep(0.01)

cap.release()