# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import os
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # 1. APP CONFIG
# # ==========================================
# st.set_page_config(page_title="Physio AI - Supine RF", layout="wide")
# st.title("🫁 Supine Chest Assistant (Random Forest)")

# # IMPORTANT: Ensure 'pose_landmarker_full.task' is in the same folder as this script!
# MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")

# # ==========================================
# # 2. ASSET LOADING (Random Forest)
# # ==========================================
# @st.cache_resource
# def load_rf_assets():
#     # Load your Random Forest pkl files
#     model = joblib.load("sleep_front_model.pkl")
#     encoder = joblib.load("sleep_front_label_encoder.pkl")
#     return model, encoder

# model, encoder = load_rf_assets()

# # Skeleton mapping for the chest/torso
# TORSO_SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24)]

# # ==========================================
# # 3. UI SETUP
# # ==========================================
# st.sidebar.header("📷 Camera Settings")
# camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2, 3], index=1)
# run_app = st.sidebar.checkbox("Start Diagnostic Scan", value=True)
# frame_placeholder = st.empty()

# # ==========================================
# # 4. MAIN RUN LOOP
# # ==========================================
# if run_app:
#     # --- MEDIA PIPE INITIALIZATION (Buffer Method) ---
#     try:
#         with open(MODEL_NAME, 'rb') as f:
#             model_buffer = f.read()
        
#         base_options = python.BaseOptions(model_asset_buffer=model_buffer)
#         options = vision.PoseLandmarkerOptions(
#             base_options=base_options,
#             running_mode=vision.RunningMode.VIDEO,
#             num_poses=1,
#             min_pose_detection_confidence=0.5,
#             min_pose_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         detector = vision.PoseLandmarker.create_from_options(options)
#     except FileNotFoundError:
#         st.error(f"❌ '{MODEL_NAME}' missing! Move it into this folder: {os.getcwd()}")
#         st.stop()
#     except Exception as e:
#         st.error(f"❌ MediaPipe Error: {e}")
#         st.stop()

#     # Camera Setup
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero Lag for DroidCam
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret: break

#             # 1. Mirroring Fix: display is mirrored, AI sees raw frame
#             display_frame = cv2.flip(frame, 1)
#             raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, _ = display_frame.shape
            
#             # 2. Process MediaPipe
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
#             timestamp_ms = int(time.time() * 1000)
#             result = detector.detect_for_video(mp_image, timestamp_ms)

#             checklist = []
#             is_ready = False
#             status_color = (150, 150, 150) # Neutral Gray

#             # HUD UI
#             overlay = display_frame.copy()
#             cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
#             cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)

#             if result.pose_landmarks:
#                 landmarks = result.pose_landmarks[0]
                
#                 # PREDICTION DATA (Raw 99 coordinates)
#                 raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                
#                 # Predict with Random Forest
#                 df = pd.DataFrame([raw_coords.flatten()])
#                 pred = model.predict(df)[0]
#                 label = encoder.inverse_transform([pred])[0]

#                 # MIRROR DRAWING: Flip X-coords for skeleton lines
#                 pixel_pts = [(int((1 - p[0]) * w), int(p[1] * h)) for p in raw_coords]

#                 # CLINICAL LOGIC
#                 # Check for shoulder level (y-axis)
#                 shoulder_diff = abs(landmarks[11].y - landmarks[12].y)
#                 is_level = shoulder_diff < 0.03
                
#                 if is_level:
#                     checklist.append("[V] SHOULDERS LEVEL")
#                 else:
#                     checklist.append("[X] LEVEL SHOULDERS")

#                 # State Verification
#                 if label == "Correct_Sleep_Front" and is_level:
#                     checklist.append("[V] SUPINE POSE READY")
#                     is_ready = True
#                     status_color = (0, 255, 0) # Clinical Green
#                 else:
#                     checklist.append("[X] ALIGNING TORSO...")
#                     status_color = (0, 0, 255) # Warning Red

#                 # Draw Skeleton
#                 for conn in TORSO_SKELETON:
#                     cv2.line(display_frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
#                 for idx in [11, 12, 23, 24]:
#                     cv2.circle(display_frame, pixel_pts[idx], 8, (255, 255, 255), -1)
#             else:
#                 checklist = ["[X] NO PATIENT DETECTED"]

#             # Final Renders
#             if is_ready:
#                 cv2.rectangle(display_frame, (0,0), (w,h), (0, 255, 0), 12)
#                 cv2.putText(display_frame, "STABLE - READY FOR CAPTURE", (w//2 - 280, 80), 
#                             cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

#             for i, msg in enumerate(checklist):
#                 color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
#                 cv2.putText(display_frame, msg, (45, h - 120 + (i*50)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

#     finally:
#         cap.release()
#         if 'detector' in locals():
#             detector.close()
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
# 1. OPTIMIZATION & SMOOTHING UTILITIES
# ==========================================
class PoseStabilizer:
    def __init__(self, alpha=0.2): 
        self.alpha = alpha
        self.prev_l = None

    def smooth(self, current_l):
        if self.prev_l is None:
            self.prev_l = current_l
            return current_l
        smoothed = []
        for p, c in zip(self.prev_l, current_l):
            s_pt = type(c)(
                x = p.x * (1 - self.alpha) + c.x * self.alpha,
                y = p.y * (1 - self.alpha) + c.y * self.alpha,
                z = p.z * (1 - self.alpha) + c.z * self.alpha,
                visibility = c.visibility
            )
            smoothed.append(s_pt)
        self.prev_l = smoothed
        return smoothed

def normalize_landmarks(landmarks):
    # Centering coordinates on the hips for stability
    base_x = (landmarks[23].x + landmarks[24].x) / 2
    base_y = (landmarks[23].y + landmarks[24].y) / 2
    return [coord for lm in landmarks for coord in [lm.x - base_x, lm.y - base_y, lm.z]]

# ==========================================
# 2. UI & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Axoris Radiology AI", layout="wide")
st.title("🛡️ Axoris: Clinical Supine Instructor")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
RF_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sleep_front/sleep_front_model.pkl").replace("\\", "/")
ENCODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/sleep_front/sleep_front_label_encoder.pkl").replace("\\", "/")

@st.cache_resource
def load_assets():
    try:
        return joblib.load(RF_MODEL), joblib.load(ENCODER)
    except:
        st.error("❌ Model or Encoder files (.pkl) missing!")
        return None, None

model, encoder = load_assets()
stabilizer = PoseStabilizer(alpha=0.25) # Stops the flicker

# ==========================================
# 3. CAMERA & SIDEBAR
# ==========================================
st.sidebar.header("📷 Control Panel")
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 2], format_func=lambda x: "Laptop Camera" if x==0 else "Droid Camera", index=0)
run_app = st.sidebar.toggle("Start Analysis", value=True)

col1, col2 = st.columns([2.5, 1])
with col2:
    st.subheader("📢 Clinical Instructions")
    instr_box = st.empty()
    st.divider()
    p_bar = st.progress(0)
    p_txt = st.empty()

with col1:
    frame_window = st.empty()

# ==========================================
# 4. ROBUST RUN LOOP
# ==========================================
if run_app and model:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.8 # Higher tracking for less flicker
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Zero lag for DroidCam
    
    if not cap.isOpened():
        st.error(f"❌ Cannot open camera {camera_index}. Please check DroidCam connection.")
        st.stop()
        
    frame_count = 0
    last_conf = 0
    last_l = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if 'active_stop_event' in globals() and active_stop_event.is_set(): break
            import time
            time.sleep(0.01)
            continue
        
        frame_count += 1
        # Mirroring for the user
        display_frame = cv2.flip(frame, 1)
        h, w, _ = display_frame.shape

        # 🔥 Speed Trick: Run AI every 2nd frame
        if frame_count % 2 == 0:
            small_rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
            current_timestamp_ms = int(time.time() * 1000)
            if \'last_timestamp_ms\' not in locals(): last_timestamp_ms = 0
            if current_timestamp_ms <= last_timestamp_ms: current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            
            if result.pose_landmarks:
                # 1. Smooth landmarks
                last_l = stabilizer.smooth(result.pose_landmarks[0])
                # 2. Predict with 50% Threshold
                feat = normalize_landmarks(last_l)
                probs = model.predict_proba([feat])[0]
                c_idx = np.where(encoder.classes_ == 'Correct_Sleep_Front')[0][0]
                last_conf = probs[c_idx]

        # --- RENDERING (Smooth & Clear) ---
        if last_l:
            is_ready = last_conf >= 0.50
            
            # --- STRICT GEOMETRIC CHECKS ---
            # 1. Facing Camera (Nose visible, left shoulder to the right of right shoulder in raw coords)
            nose_vis = last_l[0].visibility > 0.5
            is_facing_front = last_l[11].x > last_l[12].x
            
            # 2. Hands Straight Down (Wrists must be near or past the hips along the body axis)
            shoulder_center_y = (last_l[11].y + last_l[12].y) / 2
            hip_center_y = (last_l[23].y + last_l[24].y) / 2
            
            # Assuming vertical orientation in frame, wrists y should be >= hips y
            # (In mediapipe, y=0 is top, y=1 is bottom)
            left_hand_straight = last_l[15].y > last_l[23].y - 0.05
            right_hand_straight = last_l[16].y > last_l[24].y - 0.05
            hands_straight = left_hand_straight and right_hand_straight
            
            # 3. Shoulders Level
            shoulder_tilt = abs(last_l[11].y - last_l[12].y)
            is_level = shoulder_tilt < 0.05
            
            # 4. Enforce "Laying Down" vs "Standing" via ML model confidence + geometric constraints
            # The RF model (last_conf) distinguishes standing vs lying down.
            # We strictly override it if the geometric constraints fail.
            if not (nose_vis and is_facing_front and hands_straight and is_level):
                is_ready = False
                
            color = (0, 255, 0) if is_ready else (0, 0, 255)
            
            # Clinical Logic (Mirroring Adjusted)
            pixel_pts = [(int((1 - p.x) * w), int(p.y * h)) for p in last_l]
            
            # Draw Skeleton
            for s, e in [(11, 12), (23, 24), (11, 23), (12, 24)]:
                cv2.line(display_frame, pixel_pts[s], pixel_pts[e], color, 4, cv2.LINE_AA)
            
            # Face Crosshair
            cv2.line(display_frame, pixel_pts[9], pixel_pts[10], (255, 200, 0), 2, cv2.LINE_AA)

            # Update Instructions
            if is_ready:
                instr_box.success("✅ SUPINE POSE READY: HOLD STILL")
                cv2.rectangle(display_frame, (0,0), (w,h), (0, 255, 0), 10)
            else:
                if not (nose_vis and is_facing_front):
                    instr_box.error("⚠️ ERROR: FACE CAMERA (NO STOMACH POSE)")
                elif not hands_straight:
                    instr_box.error("⚠️ ERROR: KEEP HANDS STRAIGHT DOWN")
                elif not is_level:
                    instr_box.error("⚠️ ERROR: LEVEL YOUR SHOULDERS")
                elif last_conf < 0.50:
                    instr_box.error("⚠️ ERROR: LAY DOWN PROPERLY (NOT STANDING)")
                else:
                    instr_box.error("⚠️ ERROR: REALIGN TORSO")

            p_bar.progress(float(last_conf))
            p_txt.write(f"Precision Score: {last_conf*100:.1f}%")

        # -- INJECT TELEMETRY FOR FRONTEND --
        if 'global_telemetry' in globals():
            if last_l:
                is_ready = last_conf >= 0.50
                nose_vis = last_l[0].visibility > 0.5
                is_facing_front = last_l[11].x > last_l[12].x
                left_hand_straight = last_l[15].y > last_l[23].y - 0.05
                right_hand_straight = last_l[16].y > last_l[24].y - 0.05
                hands_straight = left_hand_straight and right_hand_straight
                shoulder_tilt = abs(last_l[11].y - last_l[12].y)
                is_level = shoulder_tilt < 0.05
                
                if not (nose_vis and is_facing_front and hands_straight and is_level):
                    is_ready = False
                
                if is_ready:
                    global_telemetry['message'] = "Perfect alignment. Keep holding."
                    global_telemetry['accuracy'] = int(last_conf * 100) if last_conf > 0.95 else 95
                    global_telemetry['status'] = "good"
                else:
                    if not (nose_vis and is_facing_front):
                        global_telemetry['message'] = "Warning: FACE CAMERA (NO STOMACH POSE)"
                    elif not hands_straight:
                        global_telemetry['message'] = "Warning: KEEP HANDS STRAIGHT DOWN"
                    elif not is_level:
                        global_telemetry['message'] = "Warning: LEVEL YOUR SHOULDERS"
                    elif last_conf < 0.50:
                        global_telemetry['message'] = "Warning: LAY DOWN PROPERLY (NOT STANDING)"
                    else:
                        global_telemetry['message'] = "Warning: REALIGN TORSO"
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