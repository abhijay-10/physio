### without skeleton lines
# import cv2
# import mediapipe as mp
# import time
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from pose_utils import extract_landmarks, calculate_angle

# def run_arm_raise():
#     from mediapipe.tasks.python import vision
#     from mediapipe.tasks.python.core.base_options import BaseOptions
#     from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

#     model_path = "../../pose_landmarker_full.task"
#     detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_path=model_path),
#         running_mode=vision.RunningMode.VIDEO
#     ))

#     cap = cv2.VideoCapture(0)
#     reps, stage = 0, "down"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.flip(frame, 1)

#         mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         res = detector.detect_for_video(mp_img, int(time.time() * 1000))

#         if res.pose_landmarks:
#             lm = extract_landmarks(res.pose_landmarks[0])
#             # SHOULDER ANGLE: Hip -> Shoulder -> Elbow
#             angle = calculate_angle(lm["left_hip"], lm["left_shoulder"], lm["left_elbow"])

#             # REPS: Reset at 100, Count at 150
#             if angle < 100: stage = "down"
#             if angle > 150 and stage == "down":
#                 reps += 1
#                 stage = "up"

#         cv2.rectangle(frame, (0,0), (320, 80), (30,30,30), -1)
#         cv2.putText(frame, f"ARM RAISES: {reps}", (20, 50), 1, 2, (0,255,0), 3)
#         cv2.imshow("Arm Raise AI Interface", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__": run_arm_raise()


## with skeleton lines
import cv2
import mediapipe as mp
import time
import sys
import os

# Connect to shared pose_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pose_utils import extract_landmarks, calculate_angle

def run_arm_raise():
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

    model_path = "../../pose_landmarker_full.task"
    detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    ))

    # Define the skeleton connections (Professional Layout)
    CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16), # Arms
        (11, 12), (11, 23), (12, 24), (23, 24), # Torso
        (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
    ]

    cap = cv2.VideoCapture(0)
    
    # Initialize variables for stability
    reps = 0
    stage = "down"
    angle = 0
    feedback = "Ready"
    color = (255, 255, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_img, int(time.time() * 1000))

        if res.pose_landmarks:
            landmarks = res.pose_landmarks[0]
            lm = extract_landmarks(landmarks)
            
            # 1. CALCULATE SHOULDER ANGLE
            angle = calculate_angle(lm["left_hip"], lm["left_shoulder"], lm["left_elbow"])

            # 2. DRAW SKELETON LINES (Professional Cyan)
            for connection in CONNECTIONS:
                start_pt = landmarks[connection[0]]
                end_pt = landmarks[connection[1]]
                
                c1 = (int(start_pt.x * w), int(start_pt.y * h))
                c2 = (int(end_pt.x * w), int(end_pt.y * h))
                cv2.line(frame, c1, c2, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw Joint Points (Red/White)
            for p in landmarks:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)

            # 3. ML STATE LOGIC
            if angle < 100: 
                stage = "down"
                feedback = "Lift Arms"
                color = (255, 255, 255)
            
            if angle > 150 and stage == "down":
                reps += 1
                stage = "up"
                feedback = "Rep Counted!"
                color = (0, 255, 0)

        # --- UI OVERLAY ---
        # Dark panel for contrast
        cv2.rectangle(frame, (0, 0), (400, 150), (30, 30, 30), -1)
        
        cv2.putText(frame, f"ARM RAISES: {reps}", (20, 50), 1, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"ANGLE: {int(angle)}", (20, 95), 1, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"STATUS: {feedback}", (20, 135), 1, 1.2, color, 2, cv2.LINE_AA)

        cv2.imshow("Professional Arm Raise AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    run_arm_raise()