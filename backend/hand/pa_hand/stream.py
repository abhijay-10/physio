import os
import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import math

# -------- ASSETS & CONFIG --------
model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hand/pa_hand/pa_model.pkl").replace("\\", "/"))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pa_hand", "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(7,8),
    (5,9),(9,10),(11,12), (9,13),(13,14),(15,16),
    (13,17),(17,18),(19,20), (0,17)
]

def draw_skeleton_placeholders(frame, h, w):
    overlay = frame.copy()
    start_x = 40
    line_width = int(w * 0.75)
    for i in range(5):
        y_pos = h - 165 + (i * 32)
        cv2.line(overlay, (start_x, y_pos), (start_x + line_width, y_pos), (80, 80, 80), 6)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def calculate_joint_angle(p1, p2, p3):
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 * mag2 == 0: return 0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(val))

def get_diagnostics(landmarks):
    p5, p17 = landmarks[5], landmarks[17]
    # Symmetrical angle calculation
    dx = abs(p17.x - p5.x)
    dy = p17.y - p5.y
    current_angle = abs(math.degrees(math.atan2(dy, dx)))
    rotation_needed = int(current_angle)
    
    # Check joint straightness
    fingers_joints = [
        [2, 3, 4],                  # Thumb
        [5, 6, 7], [6, 7, 8],       # Index
        [9, 10, 11], [10, 11, 12],  # Middle
        [13, 14, 15], [14, 15, 16], # Ring
        [17, 18, 19], [18, 19, 20]  # Pinky
    ]
    straightness_pass = True
    for joints in fingers_joints:
        angle = calculate_joint_angle(landmarks[joints[0]], landmarks[joints[1]], landmarks[joints[2]])
        if angle < 165: # Strict straightness check
            straightness_pass = False
            break

    # Strict check: ALL 4 fingers must be fully extended, straight, and pointing UP
    # Tip must be above PIP, which must be above MCP (y goes down in images, so smaller y is higher)
    # ALSO: The thumb must be extended outwards (not tucked).
    extended = (
        straightness_pass and
        landmarks[8].y < landmarks[6].y < landmarks[5].y and   # Index
        landmarks[12].y < landmarks[10].y < landmarks[9].y and # Middle
        landmarks[16].y < landmarks[14].y < landmarks[13].y and # Ring
        landmarks[20].y < landmarks[18].y < landmarks[17].y and # Pinky
        math.hypot(landmarks[4].x - landmarks[5].x, landmarks[4].y - landmarks[5].y) > 0.115 # Thumb extended outward (not tucked)
    )

    return {
        "angle": abs(int(current_angle)),
        "rotation_needed": rotation_needed,
        "wrist_x": landmarks[0].x,
        "extended": extended,
        "spread": abs(landmarks[8].x - landmarks[20].x)
    }

# -------- MEDIAPIPE SETUP --------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# -------- MAIN UI --------
st.title("🩺 Professional PA Hand Radiography Assistant")
camera_index = st.selectbox("Select Camera", [0,1,2,3], index=1)
run = st.checkbox("Start Diagnostic Scan")

frame_placeholder = st.empty()
timestamp = 0

if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        instructions = []
        is_fully_correct = False
        status_color = (150, 150, 150)

        # Draw UI Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                # Get Handedness (Left or Right)
                raw_handedness = result.handedness[idx][0].category_name if result.handedness else "Unknown"
                # Swap names to account for cv2 mirror flip
                if raw_handedness == "Left":
                    handedness = "Right"
                elif raw_handedness == "Right":
                    handedness = "Left"
                else:
                    handedness = "Unknown"
                
                m = get_diagnostics(landmarks)
                
                instructions.append(f"[PASS] {handedness} Hand Detected")
                
                # 1. Centering
                instructions.append("[PASS] Hand in frame" if 0.2 < m['wrist_x'] < 0.8 else "[FAIL] Move hand to center")
                
                # 2. Rotation
                if m['angle'] < 25:
                    instructions.append(f"[PASS] Hand Flat (Tilt: {m['angle']} deg)")
                else:
                    instructions.append(f"[FAIL] Rotate hand {m['rotation_needed']} deg to flat")
                
                # 3. Extension
                instructions.append("[PASS] Fingers/Thumb extended" if m['extended'] else "[FAIL] Extend fingers & thumb")
                
                # 4. Spread
                instructions.append("[PASS] Digits spread" if m['spread'] > 0.18 else "[FAIL] Spread fingers wider")
                
                # 5. Flat on Surface (Scale check)
                aspect_ratio = w / h
                dx_length = (landmarks[0].x - landmarks[9].x) * aspect_ratio
                dy_length = landmarks[0].y - landmarks[9].y
                palm_length = math.hypot(dx_length, dy_length)
                
                scale_pass = palm_length < 0.45 and landmarks[0].y > 0.88
                if scale_pass:
                    instructions.append("[PASS] Hand flat on surface")
                else:
                    instructions.append("[FAIL] Lower hand to table surface")
                
                # FINAL VALIDATION CHECK
                geometry_pass = (0.2 < m['wrist_x'] < 0.8) and (m['angle'] < 25) and m['extended'] and (m['spread'] > 0.18) and scale_pass
                
                if geometry_pass:
                    is_fully_correct = True
                    status_color = (0, 255, 0)
                    instructions.append("[PASS] READY FOR CAPTURE")
                else:
                    is_fully_correct = False
                    status_color = (0, 0, 255)
                    instructions.append("[FAIL] POSTURE INCOMPLETE")

                # Draw Hand Skeleton
                for c in HAND_CONNECTIONS:
                    x1, y1 = int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h)
                    x2, y2 = int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h)
                    cv2.line(frame, (x1,y1), (x2,y2), status_color, 3)

        else:
            draw_skeleton_placeholders(frame, h, w)
            cv2.putText(frame, "SEARCHING FOR SENSOR...", (40, h-190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # -------- SUCCESS OVERLAY --------
        if is_fully_correct:
            # Large green banner for "Great" status
            cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 220, 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            # Add a subtle glow/border around the frame
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

        # Render Diagnostics Checklist
        for i, msg in enumerate(instructions):
            color = (0, 255, 0) if "[PASS]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 180 + (i*26)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # -- INJECT TELEMETRY FOR FRONTEND --
        if 'global_telemetry' in globals():
            global_telemetry['is_correct'] = is_fully_correct
            fail_msgs = [m for m in instructions if "[FAIL]" in m or "[X]" in m]
            if is_fully_correct:
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

        frame_placeholder.image(frame, channels="BGR")
        import time
        time.sleep(0.01) # Yield GIL
    cap.release()


# import streamlit as st
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import joblib
# import math
# import numpy as np

# # -------- ASSETS & CONFIG --------
# model = joblib.load(os.path.join(os.path.dirname(__file__), "pa_model.pkl"))
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(7,8),
#     (5,9),(9,10),(11,12), (9,13),(13,14),(15,16),
#     (13,17),(17,18),(19,20), (0,17)
# ]

# def check_all_landmarks_in_frame(landmarks):
#     """Ensures every single part of the hand is visible on screen."""
#     for lm in landmarks:
#         if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
#             return False
#     return True

# def get_diagnostics(landmarks):
#     p5, p17 = landmarks[5], landmarks[17]
#     current_angle = math.degrees(math.atan2(p17.y - p5.y, p17.x - p5.x))
#     rotation_needed = -int(current_angle) 
    
#     # Check if hand is too close or far (Z-axis)
#     avg_z = np.mean([lm.z for lm in landmarks])
    
#     return {
#         "angle": abs(int(current_angle)),
#         "rotation_needed": rotation_needed,
#         "wrist_x": landmarks[0].x,
#         "extended": landmarks[8].y < landmarks[5].y,
#         "spread": abs(landmarks[8].x - landmarks[20].x),
#         "in_frame": check_all_landmarks_in_frame(landmarks),
#         "depth": avg_z
#     }

# # -------- MEDIAPIPE SETUP --------
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=1
# )
# detector = vision.HandLandmarker.create_from_options(options)

# # -------- MAIN UI --------
# st.title("🩺 Professional PA Hand Radiography Assistant")
# camera_index = st.selectbox("Select Camera", [0,1,2,3], index=1)
# run = st.checkbox("Start Diagnostic Scan")

# frame_placeholder = st.empty()
# timestamp = 0

# if run:
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result = detector.detect_for_video(mp_image, timestamp)
#         timestamp += 1

#         instructions = []
#         is_fully_correct = False
#         status_color = (150, 150, 150)

#         # Draw UI Overlay Box
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, h-220), (w, h), (15, 15, 15), -1)
#         cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

#         if result.hand_landmarks:
#             for landmarks in result.hand_landmarks:
#                 m = get_diagnostics(landmarks)
                
#                 # ML Prediction with Reshape
#                 data_row = np.array([
#                     landmarks[0].z, landmarks[8].z, landmarks[12].z, 
#                     landmarks[16].z, landmarks[20].z, 
#                     abs(landmarks[2].x - landmarks[17].x), 
#                     landmarks[0].x
#                 ]).reshape(1, -1)
                
#                 ml_ready = model.predict(data_row)[0] == 1

#                 # 1. STRICT FRAME CHECK (Fixes your issue)
#                 instructions.append("[V] Hand fully in frame" if m['in_frame'] else "[X] Move hand fully into frame")
                
#                 # 2. ROTATION
#                 if m['angle'] < 10:
#                     instructions.append(f"[V] Hand Flat (Tilt: {m['angle']}deg)")
#                 else:
#                     instructions.append(f"[X] Rotate hand {m['rotation_needed']}deg to flat")
                
#                 # 3. EXTENSION
#                 instructions.append("[V] Fingers extended" if m['extended'] else "[X] Straighten fingers")
                
#                 # 4. SPREAD
#                 instructions.append("[V] Digits spread" if m['spread'] > 0.15 else "[X] Spread fingers wider")
                
#                 # 5. FINAL STATUS (Must meet ALL conditions)
#                 if ml_ready and m['in_frame'] and m['angle'] < 10:
#                     is_fully_correct = True
#                     status_color = (0, 255, 0)
#                     instructions.append("[V] STATUS: READY")
#                 else:
#                     is_fully_correct = False
#                     status_color = (0, 0, 255)
#                     instructions.append("[X] STATUS: INCOMPLETE")

#                 # Draw Skeleton
#                 for c in HAND_CONNECTIONS:
#                     x1, y1 = int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h)
#                     x2, y2 = int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h)
#                     cv2.line(frame, (x1,y1), (x2,y2), status_color, 3)

#         else:
#             instructions = ["[X] NO HAND DETECTED"] * 5

#         # -------- SUCCESS OVERLAY --------
#         if is_fully_correct:
#             cv2.putText(frame, "GREAT! CORRECT POSTURE", (w//2 - 220, 100), 
#                         cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
#             cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)

#         # Render Instructions
#         for i, msg in enumerate(instructions):
#             color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
#             cv2.putText(frame, msg, (45, h - 160 + (i*32)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         frame_placeholder.image(frame, channels="BGR")
#     cap.release()