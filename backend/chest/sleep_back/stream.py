import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. EINSTEIN STABILIZER (Alpha Smoothing)
# ==========================================
class PoseStabilizer:
    def __init__(self, alpha=0.15):
        """Lower alpha = more stable, Higher = more responsive."""
        self.alpha = alpha
        self.prev_l = None

    def smooth(self, current_l):
        if self.prev_l is None:
            self.prev_l = current_l
            return current_l
        
        smoothed = []
        for p, c in zip(self.prev_l, current_l):
            smoothed_pt = type(c)(
                x = p.x * (1 - self.alpha) + c.x * self.alpha,
                y = p.y * (1 - self.alpha) + c.y * self.alpha,
                z = p.z * (1 - self.alpha) + c.z * self.alpha,
                visibility = c.visibility
            )
            smoothed.append(smoothed_pt)
        self.prev_l = smoothed
        return smoothed

# ==========================================
# 2. CONFIGURATION & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Physio AI Dashboard", layout="wide")
st.title("🛡️ Physio AI: Clinical Supine Analysis")

# Ensure these files are in your directory!
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
RF_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sleep_back/sleep_back_model.pkl").replace("\\", "/")
ENCODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sleep_back/sleep_back_label_encoder.pkl").replace("\\", "/")

@st.cache_resource
def load_ml_assets():
    try:
        m = joblib.load(RF_MODEL)
        e = joblib.load(ENCODER)
        return m, e
    except:
        return None, None

model, encoder = load_ml_assets()
stabilizer = PoseStabilizer(alpha=0.2) # Adjusted for stability

# Skeleton Connections
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso Box
    (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
]

# ==========================================
# 3. UI LAYOUT
# ==========================================
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📋 Clinical Checklist")
    chk_patient = st.empty()
    chk_level = st.empty()
    chk_rotation = st.empty()
    st.divider()
    st.subheader("💡 System Advice")
    advice_box = st.empty()

with col1:
    camera_idx = st.sidebar.selectbox("Select Camera Source", options=[0, 2], format_func=lambda x: "Laptop Camera" if x==0 else "Droid Camera", index=0)
    run_diagnostic = st.sidebar.toggle("Start Live Analysis", value=True)
    frame_placeholder = st.empty()

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if run_diagnostic:
    # Initialize MediaPipe Tasks
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Initialize Camera
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            if 'active_stop_event' in globals() and active_stop_event.is_set(): break
            import time
            time.sleep(0.01)
            continue
        display_frame = cv2.flip(frame, 1)
        h, w, _ = display_frame.shape
        if 'frame_count' not in locals():
            frame_count = 0
            last_px = None
            last_final_color = (0, 0, 255)
            last_status_text = "SCANNING..."
            last_is_ready = False
            last_checklist = ["[X] POSITION BACK TO CAMERA"] * 2
            
        frame_count += 1
        
        if frame_count % 2 == 0:
            small_frame = cv2.resize(frame, (256, 256))
            small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
            current_timestamp_ms = int(time.time() * 1000)
            if \'last_timestamp_ms\' not in locals(): last_timestamp_ms = 0
            if current_timestamp_ms <= last_timestamp_ms: current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            
            if result.pose_landmarks:
                raw_landmarks = result.pose_landmarks[0]
                l = stabilizer.smooth(raw_landmarks)
                
                torso_height = max(abs(l[23].y - l[11].y), 0.1)
                shoulder_tilt = abs(l[11].y - l[12].y) / torso_height
                z_twist = abs(l[11].z - l[12].z)
    
                features = []
                for lm in l: features.extend([lm.x, lm.y, lm.z])
                pred_idx = model.predict([features])[0]
                label = encoder.inverse_transform([pred_idx])[0]
    
                is_level = shoulder_tilt < 0.12
                is_flat = z_twist < 0.08
                is_correct = (label == "Correct_Sleep_Back")
    
                last_checklist = []
                if is_level:
                    last_checklist.append("[V] Shoulders Level")
                else:
                    last_checklist.append("[X] Shoulders Tilted")
                
                if is_flat:
                    last_checklist.append("[V] Spine Neutral")
                else:
                    last_checklist.append("[X] Spine Twisted")
    
                chk_patient.success("Patient: Detected")
                if is_level: chk_level.success("Shoulders: Level")
                else: chk_level.error("Shoulders: Tilted")
                
                if is_flat: chk_rotation.success("Spine: Neutral")
                else: chk_rotation.error("Spine: Twisted")
    
                if is_level and is_flat and is_correct:
                    last_final_color = (0, 255, 0)
                    last_status_text = "ALIGNMENT CORRECT ✅"
                    last_is_ready = True
                    advice_box.success("Posture is stable and clinically valid.")
                else:
                    last_final_color = (0, 0, 255)
                    last_status_text = "ALIGNMENT FAILED ❌"
                    last_is_ready = False
                    if not is_level: advice_box.warning("Lower your high shoulder to align with the other.")
                    elif not is_flat: advice_box.warning("Flatten your back; avoid rotating your torso.")
    
                last_px = {i: (int((1 - l[i].x) * w), int(l[i].y * h)) for i in range(33)}
            else:
                last_px = None
                last_final_color = (0, 0, 255)
                last_status_text = "SCANNING..."
                last_is_ready = False
                last_checklist = ["[X] POSITION BACK TO CAMERA"] * 2
                
        px = last_px
        final_color = last_final_color
        status_text = last_status_text
        is_ready = last_is_ready
        checklist = last_checklist

        if px:
            for s, e in SKELETON:
                cv2.line(display_frame, px[s], px[e], final_color, 3)
            for joint in [11, 12, 23, 24]:
                cv2.circle(display_frame, px[joint], 5, (255, 255, 255), -1)

        # Overlay HUD on Video
        cv2.putText(display_frame, status_text, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, final_color, 2)
        # -- INJECT TELEMETRY FOR FRONTEND --
        if 'global_telemetry' in globals():
            local_status = locals().get('is_fully_correct', locals().get('is_ready', False))
            local_msgs = locals().get('instructions', locals().get('checklist', []))
            fail_msgs = [m for m in local_msgs if "[FAIL]" in m or "[X]" in m]
            if local_status:
                global_telemetry['message'] = "Perfect alignment. Keep holding."
                global_telemetry['accuracy'] = 95
                global_telemetry['status'] = "good"
            elif fail_msgs:
                global_telemetry['message'] = fail_msgs[0].replace("[FAIL] ", "Warning: ").replace("[X] ", "Warning: ")
                global_telemetry['accuracy'] = 45
                global_telemetry['status'] = "bad"
            else:
                global_telemetry['message'] = "Analyzing..."
                global_telemetry['accuracy'] = 10
                global_telemetry['status'] = "calibrating"
        import time
        time.sleep(0.01) # Yield GIL
        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    detector.close()