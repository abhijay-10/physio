import os
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
# # st.set_page_config(page_title="Fan Lateral Tester", layout="wide")
# st.title("🖐️ Fan Lateral Posture Testing")
# st.markdown("This app detects **Left Fan Lateral** and **Right Fan Lateral** postures.")

# # ==========================================
# # SIDEBAR - CAMERA & CONTROLS
# # ==========================================
# st.sidebar.header("Testing Settings")
# camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
# run_test = st.sidebar.checkbox("Start Camera", value=True)

# # ==========================================
# # LOAD ASSETS (Cached)
# # ==========================================
# @st.cache_resource
# def load_fan_model():
#     # Loading the model you just trained
#     model = joblib.load("fanhand_model.pkl")
#     label_encoder = joblib.load("fanlabel_encoder.pkl")
    
#     # MediaPipe Setup
#     MODEL_PATH = "d:\\physio\\obliquehand\\hand_landmarker.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=1,
#         min_hand_detection_confidence=0.4,
#         min_hand_presence_confidence=0.4,
#         min_tracking_confidence=0.4
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, label_encoder, detector = load_fan_model()

# # Standard Hand Skeleton
# HAND_CONNECTIONS = [
#     (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
#     (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
#     (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # PERSISTENCE STATE (No Flickering)
# # ==========================================
# if 'test_points' not in st.session_state:
#     st.session_state.test_points = None
#     st.session_state.test_msg = "No Hand Detected"
#     st.session_state.test_color = (128, 128, 128)
#     st.session_state.test_timer = 0

# MAX_HOLD = 10 # Number of frames to hold lines during detection drops

# # ==========================================
# # MAIN LOOP
# # ==========================================
# frame_placeholder = st.empty()
# cap = cv2.VideoCapture(camera_index)

# while run_test:
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Camera index not found. Try switching settings in the sidebar.")
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
#     result = detector.detect(mp_image)

#     if result.hand_landmarks:
#         st.session_state.test_timer = MAX_HOLD
        
#         for hand_landmarks in result.hand_landmarks:
#             row = []
#             points = []
#             for lm in hand_landmarks:
#                 row.extend([lm.x, lm.y, lm.z])
#                 points.append((int(lm.x * w), int(lm.y * h)))

#             # PREDICTION
#             X_input = pd.DataFrame([row])
#             prediction = model.predict(X_input)[0]
#             label = label_encoder.inverse_transform([prediction])[0]

#             # LABEL LOGIC
#             if "Left" in label:
#                 st.session_state.test_msg = "✅ SUCCESS: LEFT FAN LATERAL"
#                 st.session_state.test_color = (0, 255, 0) # Green
#             elif "Right" in label:
#                 st.session_state.test_msg = "✅ SUCCESS: RIGHT FAN LATERAL"
#                 st.session_state.test_color = (255, 140, 0) # Orange/Cyan
#             else:
#                 st.session_state.test_msg = "❌ WRONG POSTURE"
#                 st.session_state.test_color = (0, 0, 255) # Red
            
#             st.session_state.test_points = points

#     else:
#         # Persistence check
#         if st.session_state.test_timer > 0:
#             st.session_state.test_timer -= 1
#         else:
#             st.session_state.test_points = None
#             st.session_state.test_msg = "Ready to Test..."
#             st.session_state.test_color = (150, 150, 150)

#     # DRAWING (STABLE)
#     if st.session_state.test_points:
#         # Draw Skeleton
#         for conn in HAND_CONNECTIONS:
#             cv2.line(frame, st.session_state.test_points[conn[0]], 
#                      st.session_state.test_points[conn[1]], 
#                      st.session_state.test_color, 3)
#         # Draw Points
#         for pt in st.session_state.test_points:
#             cv2.circle(frame, pt, 5, (255, 255, 255), -1)

#     # UI OVERLAY
#     cv2.putText(frame, st.session_state.test_msg, (30, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, st.session_state.test_color, 3)

#     # DISPLAY
#     frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
#     # Tiny delay to prevent Streamlit from overloading
#     time.sleep(0.01)

# cap.release()


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
# 1. STABLE INTERPOLATION (Zero Flickering)
# ==========================================
class HandStabilizer:
    def __init__(self, alpha=0.45):
        self.alpha = alpha 
        self.prev_points = None

    def stabilize(self, new_points):
        if self.prev_points is None:
            return new_points
        if self.prev_points is None:
            self.prev_points = new_points
            return new_points
        stable_points = self.prev_points * (1 - self.alpha) + new_points * self.alpha
        self.prev_points = stable_points
        return stable_points

# ==========================================
# ASSETS & CONFIG
# ==========================================
st.title("🩺 Fan Lateral Diagnostic Assistant")

@st.cache_resource
def load_fan_assets():
    # Loading your specific Fan Lateral models
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/fanlateral/fanhand_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/fanlateral/fanlabel_encoder.pkl").replace("\\", "/"))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "obliquehand", "hand_landmarker.task")
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_fan_assets()
stabilizer = HandStabilizer(alpha=0.45)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP SETUP
# ==========================================
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Live Analysis", value=True)
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "stream_debug.txt").replace("\\", "/"), "a") as f:
            f.write(f"Processed frame {frame_count}\n")

        # Normalize View
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
        status_color = (150, 150, 150) # Neutral Gray

        # UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 1. Smoothing
                raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                stable_pts = stabilizer.stabilize(raw_pts)
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # 2. ML Prediction
                df = pd.DataFrame([stable_pts.flatten()])
                pred = model.predict(df)[0]
                label = label_encoder.inverse_transform([pred])[0]

                # 3. DIAGNOSTICS (Fan Specific)
                # Wrist Verticality check (targeting ~90 deg)
                p0, p9 = stable_pts[0], stable_pts[9]
                angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))
                
                # Fan Separation check (Spread between Index and Pinky tips)
                # Use 2D Euclidean distance to handle vertical/slanted hand correctly
                spread = math.hypot(stable_pts[8][0] - stable_pts[20][0], stable_pts[8][1] - stable_pts[20][1])
                
                # Strict Finger Straightness Check
                def calculate_joint_angle(p1, p2, p3):
                    v1 = (p1[0] - p2[0], p1[1] - p2[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    mag1 = math.hypot(v1[0], v1[1])
                    mag2 = math.hypot(v2[0], v2[1])
                    if mag1 * mag2 == 0: return 0
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    return math.degrees(math.acos(val))
                
                fingers_joints = [
                    [9, 10, 11], [10, 11, 12],  # Middle
                    [13, 14, 15], [14, 15, 16], # Ring
                    [17, 18, 19], [18, 19, 20]  # Pinky
                ]
                straightness_pass = True
                for joints in fingers_joints:
                    j_angle = calculate_joint_angle(stable_pts[joints[0]], stable_pts[joints[1]], stable_pts[joints[2]])
                    if j_angle < 130:  # Relaxed straightness threshold from 140 to 130
                        straightness_pass = False
                        break
                        
                # "OK" Pose Check (Thumb and Index tips touching)
                p4, p8 = stable_pts[4], stable_pts[8]
                ok_dist = math.hypot(p4[0] - p8[0], p4[1] - p8[1])
                ok_pass = ok_dist < 0.06  # Slightly more lenient tip distance (0.05 -> 0.06)
                
                # --- CLINICAL CHECKLIST ---
                # 1. Centering
                checklist.append("[V] Hand Centered" if 0.2 < stable_pts[0][0] < 0.8 else "[X] Move hand to center")
                
                # 2. Verticality
                if 65 < angle < 115:  # Relaxed verticality threshold slightly from 70-110 to 65-115
                    checklist.append(f"[V] Vertical Alignment ({int(angle)}deg)")
                else:
                    checklist.append(f"[X] Align Wrist Vertically ({int(angle)}deg)")
                
                # 3. Fan Spread (Critical for Fan Lateral)
                checklist.append("[V] Fingers Fanned (Separated)" if spread > 0.12 else "[X] Separate fingers like a fan")

                # 4. Pose Form (OK sign + Straight outer fingers)
                if straightness_pass and ok_pass:
                    checklist.append("[V] Perfect 'OK' Fan Lateral Form")
                else:
                    if not ok_pass:
                        checklist.append("[X] Touch Thumb & Index tips")
                    if not straightness_pass:
                        checklist.append("[X] Straighten middle/ring/pinky")
                
                # 5. Sideways (Lateral) Position & Scale checks
                aspect_ratio = w / h
                dx_width = (stable_pts[5][0] - stable_pts[17][0]) * aspect_ratio
                dy_width = stable_pts[5][1] - stable_pts[17][1]
                palm_width = math.hypot(dx_width, dy_width)

                dx_length = (stable_pts[0][0] - stable_pts[9][0]) * aspect_ratio
                dy_length = stable_pts[0][1] - stable_pts[9][1]
                palm_length = math.hypot(dx_length, dy_length)

                tilt_ratio = palm_width / palm_length if palm_length > 0 else 0
                
                # A true lateral position has a smaller tilt_ratio (narrow palm seen edge-on).
                # If tilt_ratio >= 0.82 (increased from 0.76 for leniency), the hand is flat.
                # Also prevent hand from being lifted too close to the camera (uplifted).
                # A normal palm length resting on the table is typically < 0.45.
                lateral_orientation_pass = tilt_ratio < 0.82
                scale_pass = palm_length < 0.50  # Rely on palm length (distance) only, removing camera-framing height constraints
                
                if lateral_orientation_pass:
                    if scale_pass:
                        checklist.append("[V] Sideways resting on table")
                    else:
                        checklist.append("[X] Lower hand to table surface")
                else:
                    checklist.append("[X] Turn Hand Sideways (Ulnar Side Down)")
                
                # 6. Posture Match
                is_fan_label = "Fan" in label or "Lateral" in label
                
                if is_fan_label and straightness_pass and ok_pass and spread > 0.12 and 65 < angle < 115 and lateral_orientation_pass and scale_pass:
                    checklist.append(f"[V] Validated: {label}")
                    is_ready = True
                    status_color = (0, 255, 0)
                else:
                    if not straightness_pass:
                        checklist.append("[X] Incorrect Pose (Fingers Curled)")
                    elif not lateral_orientation_pass:
                        checklist.append("[X] Position Not Sideways")
                    elif not scale_pass:
                        checklist.append("[X] Hand Too Close to Camera")
                    elif not is_fan_label:
                        checklist.append(f"[X] AI Predicted: {label}")
                    else:
                        checklist.append("[X] Adjusting Posture...")
                    status_color = (0, 0, 255)

                # Draw Visual Skeleton
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for pt in pixel_pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        else:
            checklist = ["[X] POSITION HAND FOR FAN SCAN"] * 3

        # --- SUCCESS FEEDBACK ---
        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT FAN LATERAL", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Diagnostics
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 160 + (i*40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

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
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()