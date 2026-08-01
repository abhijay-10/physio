import os


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
# 1. STABLE INTERPOLATION
# ==========================================
class HandStabilizer:
    def __init__(self, alpha=0.45):
        self.alpha = alpha 
        self.prev_points = None

    def stabilize(self, new_points):
        if self.prev_points is None:
            self.prev_points = new_points
            return new_points
        stable_points = self.prev_points * (1 - self.alpha) + new_points * self.alpha
        self.prev_points = stable_points
        return stable_points

# ==========================================
# ASSETS & CONFIG
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/pa3finger/pa_finger_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/pa3finger/pa_finger_label_encoder.pkl").replace("\\", "/"))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "obliquehand", "hand_landmarker.task")
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
stabilizer = HandStabilizer(alpha=0.4)

# Target connections: Index, Middle, Ring
PA_CONNECTIONS = [
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16)
]
ALLOWED_PTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

# ==========================================
# MAIN UI
# ==========================================
st.title("🖐️ 3-Finger PA Diagnostic Suite")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Live Analysis", value=True)
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
        status_color = (150, 150, 150)

        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                stable_pts = stabilizer.stabilize(np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]))
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # --- MULTI-FINGER VALIDATION ---
                # Strict Finger Straightness Check for Index, Middle, and Ring fingers
                def calculate_joint_angle(p1, p2, p3):
                    v1 = (p1[0] - p2[0], p1[1] - p2[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    mag1 = math.hypot(v1[0], v1[1])
                    mag2 = math.hypot(v2[0], v2[1])
                    if mag1 * mag2 == 0: return 0
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    return math.degrees(math.acos(val))
                
                target_fingers_joints = [
                    [5, 6, 7], [6, 7, 8],       # Index
                    [9, 10, 11], [10, 11, 12],  # Middle
                    [13, 14, 15], [14, 15, 16], # Ring
                ]
                
                straightness_pass = True
                for joints in target_fingers_joints:
                    j_angle = calculate_joint_angle(stable_pts[joints[0]], stable_pts[joints[1]], stable_pts[joints[2]])
                    if j_angle < 169:  # Stricter threshold (165 -> 169) to detect slightly curled/uplifted fingers
                        straightness_pass = False
                        break

                # Verify that Index(8), Middle(12), and Ring(16) are all pointing UP and extended (tip < pip < knuckle)
                fingers_extended = []
                for tip, pip, knuckle in [(8, 6, 5), (12, 10, 9), (16, 14, 13)]:
                    if stable_pts[tip][1] < stable_pts[pip][1] < stable_pts[knuckle][1]:
                        fingers_extended.append(True)
                
                digit_count = len(fingers_extended)
                
                # ML Classification
                df = pd.DataFrame([stable_pts.flatten()])
                label = label_encoder.inverse_transform(model.predict(df))[0]

                # --- MULTI-FINGER SYMMETRICAL CHECKS ---
                aspect_ratio = w / h
                # 1. Palm width/length ratio for flatness (prevent sideways rotation)
                dx_width = (stable_pts[5][0] - stable_pts[17][0]) * aspect_ratio
                dy_width = stable_pts[5][1] - stable_pts[17][1]
                palm_width = math.hypot(dx_width, dy_width)

                dx_length = (stable_pts[0][0] - stable_pts[9][0]) * aspect_ratio
                dy_length = stable_pts[0][1] - stable_pts[9][1]
                palm_length = math.hypot(dx_length, dy_length)

                tilt_ratio = palm_width / palm_length if palm_length > 0 else 0
                flat_pass = tilt_ratio > 0.58
                
                # 2. Scale check (prevent hand from being raised/uplifted from the table)
                scale_pass = palm_length < 0.45 and stable_pts[0][1] > 0.88

                # 3. Vertical alignment (wrist to middle knuckle)
                p0, p9 = stable_pts[0], stable_pts[9]
                angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))
                vertical_pass = 70 < angle < 110

                # --- CHECKLIST LOGIC ---
                # 1. Count Check
                if digit_count == 3:
                    checklist.append("[V] All 3 Fingers Extended")
                else:
                    checklist.append(f"[X] Open 3 Fingers (Detected: {digit_count})")

                # 2. Separation Check
                spread = abs(stable_pts[8][0] - stable_pts[16][0])
                checklist.append("[V] Fingers Separated" if spread > 0.18 else "[X] Separate fingers wider")

                # 3. Finger Straightness
                checklist.append("[V] Fingers straight" if straightness_pass else "[X] Uncurl fingers")

                # 4. Flatness and Position Checks
                if flat_pass:
                    checklist.append("[V] Hand Flat (No Tilt)")
                else:
                    checklist.append("[X] Keep palm flat on surface")

                if scale_pass:
                    checklist.append("[V] Hand flat on surface")
                else:
                    checklist.append("[X] Lower hand to table surface")

                if vertical_pass:
                    checklist.append("[V] Vertical Alignment")
                else:
                    checklist.append("[X] Keep hand vertical")

                # 5. Posture Verification
                if label == "PA Finger" and digit_count == 3 and flat_pass and scale_pass and vertical_pass and spread > 0.18 and straightness_pass:
                    checklist.append("[V] Posture Validated")
                    is_ready = True
                    status_color = (0, 255, 0)
                else:
                    checklist.append("[X] Posture Incomplete")
                    status_color = (0, 0, 255)

                # Draw Visuals
                for conn in PA_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for idx in ALLOWED_PTS:
                    cv2.circle(frame, pixel_pts[idx], 4, (255, 255, 255), -1)
        else:
            checklist = ["[X] POSITION 3 FINGERS FOR SCAN"] * 3

        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT PA POSTURE", (w//2 - 320, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 180 + (i*24)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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