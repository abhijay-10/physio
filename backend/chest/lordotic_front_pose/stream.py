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
# 1. TORSO STABILIZER
# ==========================================
class TorsoStabilizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha 
        self.prev_pts = None

    def stabilize(self, new_pts):
        if self.prev_pts is None:
            self.prev_pts = new_pts
            return new_pts
        stable = self.prev_pts * (1 - self.alpha) + new_pts * self.alpha
        self.prev_pts = stable
        return stable

# ==========================================
# 2. ASSETS & CONFIG
# ==========================================
st.set_page_config(page_title="Physio AI - Lordotic Chest", layout="wide")
st.title("🫁 Lordotic Chest Assistant")

@st.cache_resource
def load_assets():
    # Load your trained Lordotic model and encoder
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/lordotic_front_pose/lordotic_model.pkl").replace("\\", "/"))
    label_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chest/lordotic_front_pose/lordotic_label_encoder.pkl").replace("\\", "/"))
    
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return model, label_encoder, detector

model, encoder, detector = load_assets()
stabilizer = TorsoStabilizer()

# Define the Torso connections (Shoulders 11,12 & Hips 23,24)
TORSO_SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24)]

# ==========================================
# 3. SIDEBAR HARDWARE SETTINGS
# ==========================================
st.sidebar.header("📷 Camera Settings")
camera_index = st.sidebar.selectbox("Select DroidCam/Camera", options=[0, 2], format_func=lambda x: "Laptop Camera" if x==0 else "Droid Camera", index=0)
run_app = st.sidebar.checkbox("Start Diagnostic Scan", value=True)

frame_placeholder = st.empty()

# ==========================================
# 4. MAIN LOOP
# ==========================================
if run_app:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: 
            if 'active_stop_event' in globals() and active_stop_event.is_set(): break
            import time
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        # HUD Overlay Box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-180), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ms_timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, ms_timestamp)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            stable_pts = stabilizer.stabilize(raw_pts)
            pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in stable_pts]

            # --- DIAGNOSTIC CALCULATION (Strict Front Pose) ---
            nose_vis = landmarks[0].visibility > 0.5
            is_facing_front = stable_pts[11][0] > stable_pts[12][0]
            shoulder_width = abs(stable_pts[11][0] - stable_pts[12][0])
            facing_front = nose_vis and is_facing_front and (shoulder_width > 0.15)
            
            shoulder_diff = abs(stable_pts[11][1] - stable_pts[12][1])
            shoulders_level = shoulder_diff < 0.04
            hip_diff = abs(stable_pts[23][1] - stable_pts[24][1])
            hips_level = hip_diff < 0.04
            shoulder_center_x = (stable_pts[11][0] + stable_pts[12][0]) / 2
            hip_center_x = (stable_pts[23][0] + stable_pts[24][0]) / 2
            torso_straight = abs(shoulder_center_x - hip_center_x) < 0.05
            
            # 3D Depth checks for tilt/rotation (sideways turn)
            shoulder_depth_diff = abs(stable_pts[11][2] - stable_pts[12][2])
            not_rotated = shoulder_depth_diff < 0.15  # Ensure not turned sideways
            
            # Check for forward/backward leaning
            shoulder_z_avg = (stable_pts[11][2] + stable_pts[12][2]) / 2
            hip_z_avg = (stable_pts[23][2] + stable_pts[24][2]) / 2
            lean_z_diff = abs(shoulder_z_avg - hip_z_avg)
            not_leaning = lean_z_diff < 0.20  # Relaxed threshold to allow natural straight posture
            
            # Check for hands down (not on chest/stomach)
            # Use inner torso bounds to prevent false positives when arms rest naturally at sides
            torso_inner_min_x = min(stable_pts[11][0], stable_pts[12][0]) + 0.02
            torso_inner_max_x = max(stable_pts[11][0], stable_pts[12][0]) - 0.02
            torso_max_y = max(stable_pts[23][1], stable_pts[24][1])
            
            hands_clear = True
            for point_idx in [15, 16, 19, 20]: # Only check wrists and fingers, ignore elbows
                pt_x, pt_y = stable_pts[point_idx][0], stable_pts[point_idx][1]
                # If hands are horizontally inside the chest AND vertically above the hips
                if (torso_inner_min_x < pt_x < torso_inner_max_x) and (pt_y < torso_max_y):
                    hands_clear = False
                    break
            
            is_straight = shoulders_level and hips_level and torso_straight and not_rotated and not_leaning

            checklist = []
            if facing_front:
                checklist.append("[V] Facing Forward")
            else:
                checklist.append("[X] Face camera directly")

            if is_straight:
                checklist.append("[V] Standing Straight")
            else:
                if not not_rotated:
                    checklist.append("[X] Stand flat (Do not turn)")
                elif not not_leaning:
                    checklist.append("[X] Stand straight (Do not lean forward/backward)")
                else:
                    checklist.append("[X] Stand straight & level shoulders")

            if hands_clear:
                checklist.append("[V] Hands clear")
            else:
                checklist.append("[X] Keep hands down at sides")

            if facing_front and is_straight and hands_clear:
                is_ready = True
                status_color = (0, 255, 0) # Success Green
            else:
                is_ready = False
                status_color = (0, 0, 255) # Error Red
        else:
            pixel_pts = None
            checklist = ["[X] POSITION TORSO IN FRAME"] * 3
            is_ready = False
            status_color = (150, 150, 150)

        if pixel_pts:
            # Draw Skeleton
            for conn in TORSO_SKELETON:
                cv2.line(frame, pixel_pts[conn[0]], pixel_pts[conn[1]], status_color, 4)
            for idx in [11, 12, 23, 24]: # Only render shoulders & hips
                cv2.circle(frame, pixel_pts[idx], 8, (255, 255, 255), -1)

        # --- FINAL FEEDBACK ---
        if is_ready:
            cv2.putText(frame, "GREAT! FRONT POSE READY", (w//2 - 250, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 12)

        # Render Checklist Text
        for i, msg in enumerate(checklist):
            color = (0, 255, 0) if "[V]" in msg else (0, 0, 255)
            cv2.putText(frame, msg, (45, h - 130 + (i*50)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()
    detector.close()




#### Mediapipe with solutions code

# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib
# import time
# import math

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # 1. MEDIAPIPE SOLUTIONS SETUP
# # ==========================================

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# pose_solution = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ==========================================
# # 2. TORSO STABILIZER
# # ==========================================

# class TorsoStabilizer:
#     def __init__(self, alpha=0.5):
#         self.alpha = alpha
#         self.prev_pts = None

#     def stabilize(self, new_pts):

#         if self.prev_pts is None:
#             self.prev_pts = new_pts
#             return new_pts

#         stable = self.prev_pts * (1 - self.alpha) + new_pts * self.alpha

#         self.prev_pts = stable

#         return stable

# # ==========================================
# # 3. STREAMLIT CONFIG
# # ==========================================

# st.set_page_config(
#     page_title="Physio AI - Lordotic Chest",
#     layout="wide"
# )

# st.title("🫁 Lordotic Chest Assistant")

# # ==========================================
# # 4. LOAD MODELS
# # ==========================================

# @st.cache_resource
# def load_assets():

#     # ML MODEL
#     model = joblib.load("lordotic_model.pkl")

#     # LABEL ENCODER
#     label_encoder = joblib.load(
#         "lordotic_label_encoder.pkl"
#     )

#     # MEDIAPIPE TASK MODEL
#     MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")

#     base_options = python.BaseOptions(
#         model_asset_path=MODEL_PATH
#     )

#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.VIDEO,
#         num_poses=1,
#         min_pose_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#         output_segmentation_masks=False
#     )

#     detector = vision.PoseLandmarker.create_from_options(
#         options
#     )

#     return model, label_encoder, detector

# model, encoder, detector = load_assets()

# stabilizer = TorsoStabilizer(alpha=0.5)

# # ==========================================
# # 5. TORSO CONNECTIONS
# # ==========================================

# TORSO_SKELETON = [
#     (11, 12),
#     (11, 23),
#     (12, 24),
#     (23, 24)
# ]

# # ==========================================
# # 6. SIDEBAR SETTINGS
# # ==========================================

# st.sidebar.header("📷 Camera Settings")

# camera_index = st.sidebar.selectbox(
#     "Select Camera",
#     options=[0, 1, 2, 3],
#     index=0
# )

# run_app = st.sidebar.checkbox(
#     "Start Diagnostic Scan",
#     value=True
# )

# frame_placeholder = st.empty()

# # ==========================================
# # 7. MAIN LOOP
# # ==========================================

# if run_app:

#     cap = cv2.VideoCapture(
#         camera_index,
#         cv2.CAP_DSHOW
#     )

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     if not cap.isOpened():
#         st.error("Cannot open camera")
#         st.stop()

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             st.error("Failed to read camera")
#             break

#         # ==========================================
#         # FRAME PREPROCESS
#         # ==========================================

#         frame = cv2.flip(frame, 1)

#         h, w, _ = frame.shape

#         rgb = cv2.cvtColor(
#             frame,
#             cv2.COLOR_BGR2RGB
#         )

#         # ==========================================
#         # MEDIAPIPE TASK API
#         # ==========================================

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb
#         )

#         ms_timestamp = int(time.time() * 1000)

#         result = detector.detect_for_video(
#             mp_image,
#             ms_timestamp
#         )

#         # ==========================================
#         # MEDIAPIPE SOLUTIONS API
#         # ==========================================

#         solution_results = pose_solution.process(rgb)

#         checklist = []

#         is_ready = False

#         status_color = (150, 150, 150)

#         # ==========================================
#         # HUD OVERLAY
#         # ==========================================

#         overlay = frame.copy()

#         cv2.rectangle(
#             overlay,
#             (0, h - 180),
#             (w, h),
#             (15, 15, 15),
#             -1
#         )

#         cv2.addWeighted(
#             overlay,
#             0.85,
#             frame,
#             0.15,
#             0,
#             frame
#         )

#         # ==========================================
#         # LANDMARK DETECTION
#         # ==========================================

#         if (
#             result.pose_landmarks and
#             solution_results.pose_landmarks
#         ):

#             # ==========================================
#             # TASK API LANDMARKS
#             # ==========================================

#             task_landmarks = result.pose_landmarks[0]

#             # ==========================================
#             # SOLUTIONS LANDMARKS
#             # ==========================================

#             solution_landmarks = (
#                 solution_results.pose_landmarks.landmark
#             )

#             # ==========================================
#             # COMBINE BOTH DETECTIONS
#             # ==========================================

#             raw_pts = []

#             for i in range(len(task_landmarks)):

#                 tx = task_landmarks[i].x
#                 ty = task_landmarks[i].y
#                 tz = task_landmarks[i].z

#                 sx = solution_landmarks[i].x
#                 sy = solution_landmarks[i].y
#                 sz = solution_landmarks[i].z

#                 final_x = (tx + sx) / 2
#                 final_y = (ty + sy) / 2
#                 final_z = (tz + sz) / 2

#                 raw_pts.append([
#                     final_x,
#                     final_y,
#                     final_z
#                 ])

#             raw_pts = np.array(raw_pts)

#             # ==========================================
#             # STABILIZATION
#             # ==========================================

#             stable_pts = stabilizer.stabilize(
#                 raw_pts
#             )

#             # ==========================================
#             # PIXEL CONVERSION
#             # ==========================================

#             pixel_pts = [
#                 (
#                     int(p[0] * w),
#                     int(p[1] * h)
#                 )
#                 for p in stable_pts
#             ]

#             # ==========================================
#             # ML PREDICTION
#             # ==========================================

#             df = pd.DataFrame([
#                 stable_pts.flatten()
#             ])

#             pred = model.predict(df)[0]

#             label = encoder.inverse_transform(
#                 [pred]
#             )[0]

#             # ==========================================
#             # DIAGNOSTIC CALCULATIONS
#             # ==========================================

#             shoulder_diff = abs(
#                 stable_pts[11][1] -
#                 stable_pts[12][1]
#             )

#             shoulders_level = shoulder_diff < 0.04

#             # Backward lean detection
#             lean_detected = (
#                 stable_pts[11][2] <
#                 stable_pts[23][2]
#             )

#             # Spine angle
#             shoulder_center = (
#                 stable_pts[11] +
#                 stable_pts[12]
#             ) / 2

#             hip_center = (
#                 stable_pts[23] +
#                 stable_pts[24]
#             ) / 2

#             dx = shoulder_center[0] - hip_center[0]
#             dy = shoulder_center[1] - hip_center[1]

#             spine_angle = abs(
#                 math.degrees(
#                     math.atan2(dy, dx)
#                 ) + 90
#             )

#             # ==========================================
#             # CHECKLIST LOGIC
#             # ==========================================

#             if shoulders_level:
#                 checklist.append(
#                     "[✓] Shoulders Level"
#                 )
#             else:
#                 checklist.append(
#                     "[X] Level your shoulders"
#                 )

#             if spine_angle < 12:
#                 checklist.append(
#                     "[✓] Spine Alignment Good"
#                 )
#             else:
#                 checklist.append(
#                     "[X] Correct Spine Alignment"
#                 )

#             if label == "Correct_Lordotic":

#                 checklist.append(
#                     "[✓] Lordotic Lean Detected"
#                 )

#                 if shoulders_level:
#                     is_ready = True
#                     status_color = (0, 255, 0)

#             else:

#                 checklist.append(
#                     "[X] Lean backward properly"
#                 )

#                 status_color = (0, 0, 255)

#             # ==========================================
#             # CUSTOM TORSO DRAWING
#             # ==========================================

#             for conn in TORSO_SKELETON:

#                 cv2.line(
#                     frame,
#                     pixel_pts[conn[0]],
#                     pixel_pts[conn[1]],
#                     status_color,
#                     4
#                 )

#             for idx in [11, 12, 23, 24]:

#                 cv2.circle(
#                     frame,
#                     pixel_pts[idx],
#                     8,
#                     (255, 255, 255),
#                     -1
#                 )

#             # ==========================================
#             # FULL MEDIAPIPE SKELETON
#             # ==========================================

#             mp_drawing.draw_landmarks(
#                 frame,
#                 solution_results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(
#                     color=(0, 255, 255),
#                     thickness=2,
#                     circle_radius=2
#                 ),
#                 mp_drawing.DrawingSpec(
#                     color=(255, 0, 255),
#                     thickness=2
#                 )
#             )

#         else:

#             checklist = [
#                 "[X] POSITION BODY IN FRAME",
#                 "[X] DETECTION FAILED"
#             ]

#         # ==========================================
#         # FINAL FEEDBACK
#         # ==========================================

#         if is_ready:

#             cv2.putText(
#                 frame,
#                 "GREAT! LORDOTIC POSITION READY",
#                 (w // 2 - 320, 80),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 1.1,
#                 (0, 255, 0),
#                 3
#             )

#             cv2.rectangle(
#                 frame,
#                 (0, 0),
#                 (w, h),
#                 (0, 255, 0),
#                 10
#             )

#         # ==========================================
#         # RENDER CHECKLIST
#         # ==========================================

#         for i, msg in enumerate(checklist):

#             color = (
#                 (0, 255, 0)
#                 if "[✓]" in msg
#                 else (0, 0, 255)
#             )

#             cv2.putText(
#                 frame,
#                 msg,
#                 (45, h - 130 + (i * 45)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 color,
#                 2
#             )

#         # ==========================================
#         # SHOW FRAME
#         # ==========================================

#         frame_placeholder.image(
#             cv2.cvtColor(
#                 frame,
#                 cv2.COLOR_BGR2RGB
#             ),
#             channels="RGB"
#         )

#     # ==========================================
#     # CLEANUP
#     # ==========================================

#     cap.release()