import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. PERFORMANCE UTILITIES
# ==========================================
def normalize_landmarks(landmarks):
    # Stabilizes detection by centering math on the hips
    base_x = (landmarks[23].x + landmarks[24].x) / 2
    base_y = (landmarks[23].y + landmarks[24].y) / 2
    return [coord for lm in landmarks for coord in [lm.x - base_x, lm.y - base_y, lm.z]]

# ==========================================
# 2. UI & ASSETS
# ==========================================
st.set_page_config(page_title="Axoris Radiology AI", layout="wide")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
RF_MODEL, ENCODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sitting_front_pose/chest_ap_rf_model.pkl").replace("\\", "/"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sitting_front_pose/chest_ap_label_encoder.pkl").replace("\\", "/")

@st.cache_resource
def load_assets():
    try: return joblib.load(RF_MODEL), joblib.load(ENCODER)
    except: return None, None

model, encoder = load_assets()

st.title("🛡️ Axoris: Clinical AP Chest Instructor")
st.markdown("---")

col1, col2 = st.columns([2.5, 1])
with col2:
    st.subheader("📢 Patient Instructions")
    instruction_box = st.empty()
    st.divider()
    st.subheader("📊 Signal Precision")
    p_bar = st.progress(0)
    p_txt = st.empty()

with col1:
    frame_window = st.empty()

# ==========================================
# 3. TURBO INSTRUCTION ENGINE
# ==========================================
if model:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.6
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    camera_index = st.sidebar.selectbox("Select Camera", options=[0, 2], format_func=lambda x: "Laptop Camera" if x==0 else "Droid Camera", index=0)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero-latency buffer

    frame_count = 0
    last_l = None 
    last_conf = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            if 'active_stop_event' in globals() and active_stop_event.is_set(): break
            import time
            time.sleep(0.01)
            continue
        
        frame_count += 1
        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        # Run AI every 2nd frame for 2x speed boost
        if frame_count % 2 == 0:
            small_frame = cv2.resize(frame, (256, 256)) # Higher resolution for better ML
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            
            current_timestamp_ms = int(time.time() * 1000)
            if \'last_timestamp_ms\' not in locals(): last_timestamp_ms = 0
            if current_timestamp_ms <= last_timestamp_ms: current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            
            if result.pose_landmarks:
                last_l = result.pose_landmarks[0]
                feat = normalize_landmarks(last_l)
                probs = model.predict_proba([feat])[0]
                c_idx = np.where(encoder.classes_ == 'Correct_AP')[0][0]
                last_conf = probs[c_idx]

        # --- DRAWING & INSTRUCTION LOGIC ---
        if last_l:
            # 50% Threshold Decision
            is_ready = last_conf >= 0.50
            color = (0, 255, 0) if is_ready else (0, 0, 255)
            
            # Instruction Logic
            shoulder_tilt = last_l[11].y - last_l[12].y
            
            if is_ready:
                instruction_box.success("✅ POSTURE CORRECT: HOLD FOR X-RAY")
                status_txt = "Great!"
            else:
                # Provide specific instructions if not ready
                if shoulder_tilt > 0.04:
                    instruction_box.error("⚠️ LOWER LEFT SHOULDER")
                elif shoulder_tilt < -0.04:
                    instruction_box.error("⚠️ LOWER RIGHT SHOULDER")
                else:
                    instruction_box.error("⚠️ REALIGN TORSO TO CENTER")
                status_txt = "ADJUSTING"

            # Render UI
            px = {i: (int(last_l[i].x * w), int(last_l[i].y * h)) for i in range(33)}
            
            # Torso Frame
            for s, e in [(11, 12), (23, 24), (11, 23), (12, 24)]:
                cv2.line(display_frame, px[s], px[e], color, 4, cv2.LINE_AA)
            
            # Mouth/Chin Alignment Crosshair
            cv2.line(display_frame, px[9], px[10], (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(display_frame, px[0], 5, (255, 255, 255), -1)

            # Update Streamlit HUD
            p_bar.progress(float(last_conf))
            p_txt.write(f"Confidence Score: {last_conf*100:.1f}%")
            cv2.putText(display_frame, status_txt, (40, 60), 1, 2, color, 3)

        # -- INJECT TELEMETRY FOR FRONTEND --
        if 'global_telemetry' in globals():
            if last_l:
                is_ready = last_conf >= 0.50
                shoulder_tilt = last_l[11].y - last_l[12].y
                if is_ready:
                    global_telemetry['message'] = "Perfect alignment. Keep holding."
                    global_telemetry['accuracy'] = int(last_conf * 100) if last_conf > 0.95 else 95
                    global_telemetry['status'] = "good"
                else:
                    if shoulder_tilt > 0.04:
                        global_telemetry['message'] = "Warning: LOWER LEFT SHOULDER"
                    elif shoulder_tilt < -0.04:
                        global_telemetry['message'] = "Warning: LOWER RIGHT SHOULDER"
                    else:
                        global_telemetry['message'] = "Warning: REALIGN TORSO TO CENTER"
                    global_telemetry['accuracy'] = max(int(last_conf * 100), 10)
                    global_telemetry['status'] = "bad"
            else:
                global_telemetry['message'] = "Warning: POSITION TORSO IN FRAME"
                global_telemetry['accuracy'] = 45
                global_telemetry['status'] = "calibrating"

        import time
        time.sleep(0.01) # Yield GIL to background camera thread
        frame_window.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    detector.close()