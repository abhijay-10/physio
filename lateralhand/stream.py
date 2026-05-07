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
# st.set_page_config(page_title="Stable Lateral Hand Detection", layout="wide")
st.title("🖐️ Stable Lateral Hand Posture Detection")

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.selectbox("Select Camera Index", options=[0, 1, 2], index=0)
run_app = st.sidebar.checkbox("Start Detection", value=True)

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("lateralhand/lateral_dual_hand_model.pkl")
    label_encoder = joblib.load("lateralhand/lateral_dual_label_encoder.pkl")

    MODEL_PATH = "d:\\physio\\obliquehand\\hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4, # Lowered slightly to prevent drops
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# PERSISTENCE LOGIC (The Anti-Flicker Secret)
# ==========================================
if 'points_memory' not in st.session_state:
    st.session_state.points_memory = None
    st.session_state.msg_memory = "No Hand Detected"
    st.session_state.color_memory = (128, 128, 128)
    st.session_state.hold_timer = 0

# Increase this number to keep lines on screen longer during flickers
MAX_HOLD_FRAMES = 10 

# ==========================================
# MAIN LOOP
# ==========================================
frame_placeholder = st.empty()
cap = cv2.VideoCapture(camera_index)

while run_app:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        # 1. We found a hand! Reset the timer to maximum
        st.session_state.hold_timer = MAX_HOLD_FRAMES
        
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))

            # 2. Get the Prediction
            X = pd.DataFrame([row])
            pred = model.predict(X)[0]
            label = label_encoder.inverse_transform([pred])[0]

            # 3. Store results in memory
            if label == "Left":
                st.session_state.msg_memory = "✅ Correct: LEFT LATERAL"
                st.session_state.color_memory = (0, 255, 0)
            elif label == "Right":
                st.session_state.msg_memory = "✅ Correct: RIGHT LATERAL"
                st.session_state.color_memory = (255, 165, 0)
            else:
                st.session_state.msg_memory = "❌ Wrong Posture"
                st.session_state.color_memory = (0, 0, 255)
            
            st.session_state.points_memory = points
    else:
        # 4. Hand lost? Decrease the timer instead of deleting lines
        if st.session_state.hold_timer > 0:
            st.session_state.hold_timer -= 1
        else:
            # 5. Timer hit zero? Now we can clear the screen
            st.session_state.points_memory = None
            st.session_state.msg_memory = "Searching for Hand..."
            st.session_state.color_memory = (100, 100, 100)

    # ======================================
    # DRAWING FROM MEMORY
    # ======================================
    if st.session_state.points_memory:
        # Draw the skeleton lines
        for conn in HAND_CONNECTIONS:
            cv2.line(frame, st.session_state.points_memory[conn[0]], 
                     st.session_state.points_memory[conn[1]], 
                     st.session_state.color_memory, 3)
        # Draw the joint points
        for pt in st.session_state.points_memory:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)

    # Display status text
    cv2.putText(frame, st.session_state.msg_memory, (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, st.session_state.color_memory, 3)

    # Send to Streamlit
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    # Minimal sleep to help the browser keep up
    time.sleep(0.01)

cap.release()