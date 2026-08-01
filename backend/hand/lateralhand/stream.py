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
# 1. LANDMARK SMOOTHING
# ==========================================
class LandmarkSmoother:
    def __init__(self, window_size=2): # Reduced window size from 8 to 2 to eliminate lag
        self.window_size = window_size
        self.history = []

    def smooth(self, new_landmarks):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in new_landmarks])
        self.history.append(coords)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

# ==========================================
# ASSETS & CONFIG
# ==========================================
st.title("✋ Clinical Lateral Hand Assistant")

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/lateralhand/lateral_dual_hand_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/lateralhand/lateral_dual_label_encoder.pkl").replace("\\", "/"))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "obliquehand", "hand_landmarker.task")
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_assets()
smoother = LandmarkSmoother(window_size=2) 

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP
# ==========================================
camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
run_app = st.sidebar.checkbox("Start Scan", value=True)

frame_placeholder = st.empty()
cap = cv2.VideoCapture(camera_index)

while run_app:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    ms_timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, ms_timestamp)
    
    checklist = []
    is_fully_correct = False
    status_color = (150, 150, 150) # Default grey
    header_text = "ALIGNING..."

    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            smoothed = smoother.smooth(hand_landmarks)
            pts = [(int(c[0] * w), int(c[1] * h)) for c in smoothed]
            
            # --- STRICT DIAGNOSTIC CHECKS ---
            
            # 1. Centering (Use Middle Finger MCP (9) instead of Wrist (0) to avoid bottom-edge cutoffs)
            if 0.15 < smoothed[9][0] < 0.85 and 0.1 < smoothed[9][1] < 0.9:
                checklist.append("[V] Hand Centered")
                centered = True
            else:
                checklist.append("[X] Move hand to center")
                centered = False

            # 2. Strict Finger Straightness (Exclude Thumb for strict 165deg, check thumb separately)
            def calculate_joint_angle(p1, p2, p3):
                v1 = (p1[0] - p2[0], p1[1] - p2[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                mag1 = math.hypot(v1[0], v1[1])
                mag2 = math.hypot(v2[0], v2[1])
                if mag1 * mag2 == 0: return 0
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                return math.degrees(math.acos(val))
            
            # In a true lateral view, the middle and ring fingers are occluded and their points may jitter or collapse.
            # We only strictly check the straightness of the Index and Pinky fingers to avoid false curl detections.
            fingers_joints = [
                [5, 6, 7], [6, 7, 8],       # Index
                [17, 18, 19], [18, 19, 20]  # Pinky
            ]
            straightness_pass = True
            for joints in fingers_joints:
                if calculate_joint_angle(smoothed[joints[0]], smoothed[joints[1]], smoothed[joints[2]]) < 165:
                    straightness_pass = False
                    break
            
            if straightness_pass:
                checklist.append("[V] Fingers strictly straight")
            else:
                checklist.append("[X] Uncurl/straighten fingers")

            # 3. Superimposition / Tilt (Diagonally sideways / Half sideways)
            # Calculate palm width (Index MCP to Pinky MCP) and palm length (Wrist to Middle MCP)
            palm_width = math.hypot(smoothed[5][0] - smoothed[17][0], smoothed[5][1] - smoothed[17][1])
            palm_length = math.hypot(smoothed[0][0] - smoothed[9][0], smoothed[0][1] - smoothed[9][1])
            tilt_ratio = palm_width / palm_length if palm_length > 0 else 0

            # Ratio > 0.50 means the hand is flat (facing camera, like PA or backward hand)
            # Ratio < 0.3 means the hand is completely sideways (karate chop)
            # 0.3 <= Ratio <= 0.50 means diagonally sideways (the correct lateral posture angle)
            if 0.3 <= tilt_ratio <= 0.50:
                checklist.append("[V] Diagonal Sideways")
                lateral_pass = True
            elif tilt_ratio < 0.3:
                checklist.append("[X] Too sideways! Tilt slightly towards camera")
                lateral_pass = False
            else:
                checklist.append("[X] Flat Hand Detected! Tilt diagonally")
                lateral_pass = False

            # 4. Thumb Extension
            # Thumb tip (4) should be significantly far from index MCP (5) in the x or y direction compared to resting.
            thumb_dist = math.hypot(smoothed[4][0] - smoothed[5][0], smoothed[4][1] - smoothed[5][1])
            if thumb_dist > 0.12:
                checklist.append("[V] Thumb extended")
                thumb_pass = True
            else:
                checklist.append("[X] Extend thumb outward")
                thumb_pass = False

            # 5. Verticality
            p0, p9 = smoothed[0], smoothed[9]
            angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))
            if 70 < angle < 110:
                checklist.append(f"[V] Vertical Alignment")
                vertical_pass = True
            else:
                checklist.append(f"[X] Keep hand vertical")
                vertical_pass = False

            # 6. ML Prediction and Handedness Validation
            df = pd.DataFrame([smoothed.flatten()])
            prediction = model.predict(df)[0]
            label = label_encoder.inverse_transform([prediction])[0]
            
            # Get Handedness (Left or Right) from MediaPipe
            raw_handedness = result.handedness[idx][0].category_name if result.handedness else "Unknown"
            # Swap names to account for cv2 mirror flip
            if raw_handedness == "Left":
                handedness = "Right"
            elif raw_handedness == "Right":
                handedness = "Left"
            else:
                handedness = "Unknown"

            if label != "Wrong" and (label == raw_handedness or label == handedness):
                checklist.append(f"[V] Validated: {label} Hand")
                hand_validated = True
            else:
                if label == "Wrong":
                    checklist.append("[X] Incorrect Pose (Wrong Orientation)")
                else:
                    checklist.append(f"[X] Expected {label} Hand (Mirror Check)")
                hand_validated = False

            # --- CHECK FINAL SYNC ---
            if centered and straightness_pass and lateral_pass and thumb_pass and vertical_pass and hand_validated:
                is_fully_correct = True
                status_color = (0, 255, 0)
                header_text = f"GREAT! CORRECT {label.upper()} LATERAL"
            else:
                is_fully_correct = False
                status_color = (0, 0, 255)
                header_text = "POSTURE INCOMPLETE"

            # Draw Smoothed Skeleton
            for conn in HAND_CONNECTIONS:
                cv2.line(frame, pts[conn[0]], pts[conn[1]], status_color, 3)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
    else:
        checklist = ["[X] NO HAND DETECTED"] * 6
        is_fully_correct = False

    # --- CLINICAL UI ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-225), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if is_fully_correct:
        cv2.putText(frame, header_text, (w//2 - 320, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)
    else:
        cv2.putText(frame, f"STATUS: {header_text}", (20, h-195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    for i, msg in enumerate(checklist):
        color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
        cv2.putText(frame, msg, (35, h - 185 + (i*28)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # -- INJECT TELEMETRY FOR FRONTEND --
    if 'global_telemetry' in globals():
        fail_msgs = [m for m in checklist if "[X]" in m]
        if is_fully_correct:
            global_telemetry['message'] = "Perfect alignment. Keep holding."
            global_telemetry['accuracy'] = 95
            global_telemetry['status'] = "good"
        elif fail_msgs:
            global_telemetry['message'] = fail_msgs[0].replace("[X] ", "Warning: ")
            global_telemetry['accuracy'] = 45
            global_telemetry['status'] = "bad"
        else:
            global_telemetry['message'] = "Analyzing..."
            global_telemetry['accuracy'] = 10
            global_telemetry['status'] = "calibrating"

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    import time
    time.sleep(0.01) # Yield GIL to prevent server starvation
    
cap.release()