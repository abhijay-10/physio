import os
### Basic right and wrong code of bilateral hand with ML feedback and instructions
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
# # st.set_page_config(page_title="Bilateral PA Guide", layout="wide")
# st.title("🖐️ Bilateral PA Hand Assistant")

# # ==========================================
# # LOAD MODELS & ASSETS
# # ==========================================
# @st.cache_resource
# def load_bilateral_assets():
#     model = joblib.load("bilateral_pa_model.pkl")
#     label_encoder = joblib.load("bilateral_label_encoder.pkl")
    
#     MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=2,
#         min_hand_detection_confidence=0.4,
#         min_hand_presence_confidence=0.4,
#         min_tracking_confidence=0.4
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, label_encoder, detector = load_bilateral_assets()

# HAND_CONNECTIONS = [
#     (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
#     (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
#     (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # SIDEBAR SETTINGS
# # ==========================================
# st.sidebar.header("Camera")
# camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
# run_app = st.sidebar.checkbox("Start Assistant", value=True)

# # ==========================================
# # UI LAYOUT
# # ==========================================
# col_vid, col_inst = st.columns([2, 1])
# with col_vid:
#     frame_placeholder = st.empty()
# with col_inst:
#     st.subheader("🛠️ Guidance Commands")
#     cmd_box = st.empty()
#     tip_box = st.empty()

# # ==========================================
# # STABILITY STATE
# # ==========================================
# if 'state' not in st.session_state:
#     st.session_state.state = {"points": [], "label": "No Hand", "timer": 0}

# MAX_PERSISTENCE = 8

# # ==========================================
# # MAIN LOOP
# # ==========================================
# cap = cv2.VideoCapture(camera_index)

# while run_app:
#     ret, frame = cap.read()
#     if not ret: break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#     result = detector.detect(mp_image)

#     current_label = "No Hand"
#     current_points = []

#     if result.hand_landmarks and len(result.hand_landmarks) == 2:
#         st.session_state.state["timer"] = MAX_PERSISTENCE
        
#         # Sort hands by X to match training data
#         sorted_hands = sorted(result.hand_landmarks, key=lambda x: x[0].x)
        
#         row = []
#         for hand in sorted_hands:
#             pts = []
#             for lm in hand:
#                 row.extend([lm.x, lm.y, lm.z])
#                 pts.append((int(lm.x * w), int(lm.y * h)))
#             current_points.append(pts)

#         # Predict
#         X_in = pd.DataFrame([row])
#         pred_idx = model.predict(X_in)[0]
#         current_label = label_encoder.inverse_transform([pred_idx])[0]
        
#         st.session_state.state["label"] = current_label
#         st.session_state.state["points"] = current_points
#     else:
#         if st.session_state.state["timer"] > 0:
#             st.session_state.state["timer"] -= 1
#         else:
#             st.session_state.state["label"] = "No Hand"
#             st.session_state.state["points"] = []

#     # --- VISUAL FEEDBACK LOGIC ---
#     overlay_msg = st.session_state.state["label"]
    
#     if overlay_msg == "Bilateral PA":
#         color = (0, 255, 0) # Green
#         cmd_box.success("✅ Perfect Alignment!")
#         tip_box.info("Both hands are parallel and flat. Hold this position.")
    
#     elif overlay_msg == "Wrong":
#         color = (0, 0, 255) # Red
#         cmd_box.error("❌ Posture Incorrect")
#         tip_box.warning("""
#         **Commands to fix:**
#         1. Place **BOTH** hands flat on the table.
#         2. Keep palms facing **DOWN**.
#         3. Align your middle fingers so they are **parallel**.
#         4. Spread your fingers slightly to avoid overlapping.
#         """)
        
#         # DRAW GHOST SKELETON (Guidance Lines)
#         # This draws a "Reference" line to show where the hands should be
#         cv2.putText(frame, "FOLLOW GHOST GUIDE", (w//4, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
#     else:
#         color = (150, 150, 150)
#         cmd_box.info("Place both hands in view...")

#     # DRAW ACTUAL SKELETON
#     if st.session_state.state["points"]:
#         for hand_pts in st.session_state.state["points"]:
#             for conn in HAND_CONNECTIONS:
#                 cv2.line(frame, hand_pts[conn[0]], hand_pts[conn[1]], color, 3)
#             for pt in hand_pts:
#                 cv2.circle(frame, pt, 4, (255, 255, 255), -1)

#     cv2.putText(frame, overlay_msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#     frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
#     time.sleep(0.01)

# cap.release()


### This below code is the instructions based hand bilateral UI
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
# 1. HIGH-SPEED DUAL STABILIZER
# ==========================================
class BilateralStabilizer:
    def __init__(self, alpha=0.5): # Increased alpha for faster response
        self.alpha = alpha 
        self.prev_hands = None

    def stabilize(self, new_hands_list):
        if self.prev_hands is None or len(new_hands_list) != len(self.prev_hands):
            self.prev_hands = new_hands_list
            return new_hands_list
        
        stable_hands = []
        for i in range(len(new_hands_list)):
            stable_pts = self.prev_hands[i] * (1 - self.alpha) + new_hands_list[i] * self.alpha
            stable_hands.append(stable_pts)
        self.prev_hands = stable_hands
        return stable_hands

# ==========================================
# ASSETS & CONFIG
# ==========================================
@st.cache_resource
def load_bilateral_assets():
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/bilateralhand/bilateral_pa_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/bilateralhand/bilateral_label_encoder.pkl").replace("\\", "/"))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "obliquehand", "hand_landmarker.task")
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, label_encoder, detector = load_bilateral_assets()
stabilizer = BilateralStabilizer()

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# MAIN LOOP
# ==========================================
st.title("🖐️ Bilateral PA Hand Assistant")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)
run = st.sidebar.checkbox("Start Diagnostic Scan", value=True)
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
        status_color = (100, 100, 100)

        # UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            # Sort hands Left-to-Right for ML consistency
            sorted_hands = sorted(result.hand_landmarks, key=lambda x: x[0].x)
            raw_hands = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in sorted_hands]
            stable_hands = stabilizer.stabilize(raw_hands)
            
            # --- 1. STRICT PHYSICAL VALIDATION ---
            fingers_straight = True
            hands_spread = True
            aligned_properly = True
            thumbs_correct = True
            vertical_pass = True

            # Alignment Check: Wrists should be roughly at the same horizontal level
            wrist_0_y = stable_hands[0][0][1]
            wrist_1_y = stable_hands[1][0][1]
            if abs(wrist_0_y - wrist_1_y) > 0.15:
                aligned_properly = False

            # Vertical alignment check (both hands must point straight up vertically)
            aspect_ratio = w / h
            p0_0, p0_9 = stable_hands[0][0], stable_hands[0][9]
            angle0 = abs(math.degrees(math.atan2(p0_0[1] - p0_9[1], (p0_0[0] - p0_9[0]) * aspect_ratio)))
            
            p1_0, p1_9 = stable_hands[1][0], stable_hands[1][9]
            angle1 = abs(math.degrees(math.atan2(p1_0[1] - p1_9[1], (p1_0[0] - p1_9[0]) * aspect_ratio)))
            
            if not (75 < angle0 < 105 and 75 < angle1 < 105):
                vertical_pass = False

            # Thumbs pointing inwards AND fully extended (not curled)
            # Left hand on screen (index 0): thumb points RIGHT (x increases)
            thumb0_tip_x, thumb0_ip_x, thumb0_mcp_x = stable_hands[0][4][0], stable_hands[0][3][0], stable_hands[0][2][0]
            if not (thumb0_tip_x > thumb0_ip_x > thumb0_mcp_x):
                thumbs_correct = False
                
            # Right hand on screen (index 1): thumb points LEFT (x decreases)
            thumb1_tip_x, thumb1_ip_x, thumb1_mcp_x = stable_hands[1][4][0], stable_hands[1][3][0], stable_hands[1][2][0]
            if not (thumb1_tip_x < thumb1_ip_x < thumb1_mcp_x):
                thumbs_correct = False

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
                [2, 3, 4],                  # Thumb
                [5, 6, 7], [6, 7, 8],       # Index
                [9, 10, 11], [10, 11, 12],  # Middle
                [13, 14, 15], [14, 15, 16], # Ring
                [17, 18, 19], [18, 19, 20]  # Pinky
            ]

            for hand in stable_hands:
                # Straight Fingers Check (Strict Angles)
                for joints in fingers_joints:
                    j_angle = calculate_joint_angle(hand[joints[0]], hand[joints[1]], hand[joints[2]])
                    if j_angle < 165:
                        fingers_straight = False
                        break
                        
                # Tip must still be above MCP
                for tip_idx in [8, 12, 16, 20]:
                    tip_y = hand[tip_idx][1]
                    mcp_y = hand[tip_idx - 3][1]
                    if not (tip_y < mcp_y):
                        fingers_straight = False
                
                # Hand Spread (Distance between index and pinky tips)
                if abs(hand[8][0] - hand[20][0]) < 0.12:
                    hands_spread = False
            
            # --- 2. ML PREDICTION (Fallback/Logging) ---
            combined_row = np.concatenate([hand.flatten() for hand in stable_hands])
            prediction = model.predict(pd.DataFrame([combined_row]))[0]
            label = label_encoder.inverse_transform([prediction])[0]

            # --- DYNAMIC INSTRUCTIONS ---
            if not aligned_properly:
                checklist.append("[FAIL] Align both hands at the same height")
            else:
                checklist.append("[PASS] Hands aligned properly")

            if not vertical_pass:
                checklist.append("[FAIL] Keep both hands vertically straight")
            else:
                checklist.append("[PASS] Hands vertically straight")

            if not fingers_straight:
                checklist.append("[FAIL] Keep all fingers perfectly straight")
            elif not hands_spread:
                checklist.append("[FAIL] Spread your fingers wider")
            elif not thumbs_correct:
                checklist.append("[FAIL] Point thumbs inwards")
            else:
                checklist.append("[PASS] Fingers straight & spread correctly")

            # Final Validation Check
            if aligned_properly and vertical_pass and fingers_straight and hands_spread and thumbs_correct:
                is_ready = True
                status_color = (0, 255, 0) # Green Success
                if label == "Bilateral PA":
                    checklist.append(f"[PASS] Posture Verified (ML + Geometry)")
                else:
                    checklist.append(f"[PASS] Posture Verified (Geometry)")
            else:
                checklist.append("[FAIL] Posture Incorrect")
                status_color = (0, 0, 255) # Red Error

            # Draw skeletons
            for hand_data in stable_hands:
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in hand_data]
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 3)
                for pt in pixel_pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)

        else:
            checklist = ["[X] PLACE BOTH HANDS FULLY OPENED"] * 2

        # --- HIGHLIGHTED SUCCESS ---
        if is_ready:
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 300, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Diagnostics
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[PASS]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 175 + (i*38)), 
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
    
    cap.release()