import os

# # ==========================================
# # RELEASE
# # ==========================================
# cap.release()

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
# 1. LANDMARK SMOOTHING
# ==========================================
class LandmarkSmoother:
    def __init__(self, window_size=2):
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
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/obliquehand/hand_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/obliquehand/label_encoder.pkl").replace("\\", "/"))
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
# SIDEBAR & UI
# ==========================================
st.sidebar.header("🎥 Hardware Setup")
cam_choice = st.sidebar.selectbox("Active Device Index", [0, 1, 2, 3], index=1)
run = st.sidebar.checkbox("Run Diagnostic Engine", value=True)

st.title("🩺 Advanced Oblique Hand Suite")
st.caption("Standardized 720p Analysis | Precision Diagnostic Checklist")

frame_placeholder = st.empty()

# ==========================================
# MAIN LOOP
# ==========================================
if run:
    cap = cv2.VideoCapture(cam_choice, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error(f"❌ Camera {cam_choice} offline.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret: break

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
            status_color = (80, 80, 80)

            # --- UI OVERLAY ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-180), (w, h), (12, 12, 12), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            if result.hand_landmarks:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    # Remove smoothing entirely to prevent points floating "outside" the hand during fast motion
                    stable_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

                    # --- CORE PARAMETERS FOR STRICT SIDEWAYS (LATERAL) ---
                    
                    # 1. Superimposition (Fingers overlapping from the side view)
                    mcp_x = [stable_pts[i][0] for i in [5, 9, 13, 17]]
                    mcp_spread = max(mcp_x) - min(mcp_x)
                    lateral_pass = mcp_spread < 0.08
                    
                    # 2. Verticality
                    p0, p9 = stable_pts[0], stable_pts[9]
                    angle = abs(math.degrees(math.atan2(p0[1] - p9[1], p0[0] - p9[0])))
                    vertical_pass = 70 < angle < 110

                    # 4. Strict Finger Straightness Check (Only Index and Pinky to allow middle occlusion)
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
                        [5, 6, 7], [6, 7, 8],       # Index
                        [17, 18, 19], [18, 19, 20]  # Pinky
                    ]
                    straightness_pass = True
                    for joints in fingers_joints:
                        j_angle = calculate_joint_angle(stable_pts[joints[0]], stable_pts[joints[1]], stable_pts[joints[2]])
                        if j_angle < 165:
                            straightness_pass = False
                            break

                    # ML Classification
                    df = pd.DataFrame([stable_pts.flatten()])
                    label = label_encoder.inverse_transform(model.predict(df))[0]

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
                        hand_validated = True
                    else:
                        hand_validated = False

                    # --- FOCUSED CHECKLIST ---
                    # Parameter 1: Vertical
                    if vertical_pass:
                        checklist.append(f"[V] Vertical Alignment")
                    else:
                        checklist.append(f"[X] Keep hand vertical")
                    
                    # Parameter 2: Sideways Superimposition
                    if lateral_pass:
                        checklist.append("[V] True Sideways (Superimposed)")
                    else:
                        checklist.append("[X] Rotate hand completely sideways")
                    
                    # Parameter 3: Side Label
                    if hand_validated:
                        checklist.append(f"[V] Validated: {label} Hand")
                    else:
                        if label == "Wrong":
                            checklist.append("[X] Incorrect Pose (Wrong Orientation)")
                        else:
                            checklist.append(f"[X] Expected {label} Hand (Mirror Check)")

                    # Parameter 4: Finger Straightness (Strict)
                    checklist.append("[V] Fingers straight" if straightness_pass else "[X] Uncurl fingers")

                    # FINAL SYNC
                    if label in ["Left", "Right"] and vertical_pass and lateral_pass and straightness_pass and hand_validated:
                        is_ready = True
                        status_color = (0, 255, 0)
                    else:
                        is_ready = False
                        status_color = (0, 0, 255)

                    # Draw Skeleton
                    for conn in HAND_CONNECTIONS:
                        cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                    for pt in pixel_pts:
                        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            else:
                for i in range(3):
                    y_pos = h - 140 + (i * 35)
                    cv2.line(frame, (45, y_pos), (int(w * 0.5), y_pos), (60, 60, 60), 4)
                checklist = ["[X] NO HAND DETECTED"] * 3

            if is_ready:
                cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 280, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
                cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

            # Render Focused Instructions
            for i, msg in enumerate(checklist):
                color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
                cv2.putText(frame, msg, (45, h - 130 + (i*35)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
            import time
            time.sleep(0.01) # Yield GIL
        
        cap.release()