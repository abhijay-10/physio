import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import math
import time
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
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/obliquethumb/oblique_thumb_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/obliquethumb/oblique_label_encoder.pkl").replace("\\", "/"))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "obliquehand", "hand_landmarker.task")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
stabilizer = HandStabilizer()

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("🩺 Precision Oblique Thumb Diagnostic")
camera_index = st.sidebar.selectbox("Camera Device", options=[0, 1, 2, 3], index=0)
run_app = st.sidebar.checkbox("Start Live Analysis", value=True)

frame_placeholder = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
if run_app:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)

        checklist = []
        checks_passed = 0 
        is_fully_correct = False
        status_color = (0, 0, 255) # Default Red (BGR)

        # Background Overlay Menu
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1) # Expanded to fit 4 elements
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                stable_pts = stabilizer.stabilize(raw_pts)
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                # --- 1. ML PREDICTION & HANDEDNESS ---
                df = pd.DataFrame([stable_pts.flatten()])
                prediction = model.predict(df)[0]
                label = label_encoder.inverse_transform([prediction])[0]
                
                # Get Handedness (Left or Right)
                raw_handedness = result.handedness[idx][0].category_name if result.handedness else "Unknown"
                # Swap names to account for cv2 mirror flip
                if raw_handedness == "Left":
                    handedness = "Right"
                elif raw_handedness == "Right":
                    handedness = "Left"
                else:
                    handedness = "Unknown"
                    
                detected_hand_text = f"{handedness} Hand"

                # --- 2. SYMMETRICAL ANGLE GUIDANCE ---
                p2, p4 = stable_pts[2], stable_pts[4]
                # Using abs(dx) makes the angle calculation identical regardless of whether 
                # the thumb points left or right across the screen.
                dy = p4[1] - p2[1]
                dx = abs(p4[0] - p2[0])
                angle = abs(math.degrees(math.atan2(dy, dx)))
                
                # --- 3. REFINED CURL LOGIC ---
                curl_dist = np.mean([abs(stable_pts[i][1] - stable_pts[0][1]) for i in [8, 12, 16, 20]])
                is_curled = curl_dist < 0.28 

                # --- BUILD SYNCED CHECKLIST ---
                
                # Display which hand is detected
                checklist.append(f"[PASS] {detected_hand_text} Detected")
                
                # Check 1: All Fingers Straightness (Strict Angle Check for no deflection)
                def get_angle(p1, p2, p3):
                    v1 = (p1[0] - p2[0], p1[1] - p2[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    mag1 = math.hypot(v1[0], v1[1])
                    mag2 = math.hypot(v2[0], v2[1])
                    if mag1 * mag2 == 0: return 0
                    val = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)))
                    return math.degrees(math.acos(val))
                
                fingers_joints = [
                    [2, 3, 4],                  # Thumb
                    [5, 6, 7], [6, 7, 8],       # Index
                    [9, 10, 11], [10, 11, 12],  # Middle
                    [13, 14, 15], [14, 15, 16], # Ring
                    [17, 18, 19], [18, 19, 20]  # Pinky
                ]
                straightness_pass = True
                for joints in fingers_joints:
                    j_angle = get_angle(stable_pts[joints[0]], stable_pts[joints[1]], stable_pts[joints[2]])
                    if j_angle < 165:
                        straightness_pass = False
                        break

                is_thumb_pointing_up = stable_pts[4][1] < stable_pts[2][1]
                
                if straightness_pass and is_thumb_pointing_up:
                    checklist.append("[PASS] All fingers straight")
                    checks_passed += 1
                else:
                    checklist.append("[FAIL] Straighten all fingers (no bending)")

                # Check 2: Thumb Alignment (Strict 30 to 55 deg range as requested)
                if 30 <= angle <= 55:
                    checklist.append(f"[PASS] Thumb aligned correctly ({int(angle)} deg)")
                    checks_passed += 1
                else:
                    checklist.append(f"[FAIL] Adjust thumb tilt ({int(angle)} deg)")

                # --- GLOBAL SYNC: TRIGGER SUCCESS COLOR ---
                if checks_passed == 2:
                    is_fully_correct = True
                    status_color = (0, 255, 0)      # Pure BGR Green
                    joint_color = (255, 255, 255)   # White joints
                else:
                    status_color = (0, 0, 255)      # Pure BGR Red
                    joint_color = (0, 165, 255)     # Orange warning dots

                # --- DRAW SKELETON (THUMB ONLY) ---
                THUMB_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4)]
                for conn in THUMB_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
                for pt_idx in [0, 1, 2, 3, 4]:
                    cv2.circle(frame, pixel_pts[pt_idx], 6, joint_color, -1)
        else:
            checklist = ["[FAIL] POSITION THUMB FOR SCAN"] * 2

        # --- HIGHLIGHTED SUCCESS FEEDBACK ---
        if is_fully_correct:
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 320, 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Instructions Text onto the dark layout overlay
        for i, msg in enumerate(checklist):
            text_color = (0, 255, 0) if "[PASS]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 185 + (i*40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

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