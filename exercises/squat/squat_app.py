import cv2
import mediapipe as mp
import time
import sys
import os

# Connect to shared pose_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pose_utils import extract_landmarks, calculate_angle

def run_pro_squat():
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

    model_path = "../../pose_landmarker_full.task"
    detector = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    ))

    cap = cv2.VideoCapture(0)
    
    # --- ML STATE MACHINE (STRICT) ---
    reps = 0
    # States: STANDING, SQUATTING, ASCENDING
    state = "STANDING" 
    
    # Accuracy Variables
    hip_start_y = 0.0  # Baseline height
    max_depth_reached = False
    feedback = "Stand Straight"
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
            
            # 1. CALCULATE CORE METRICS
            knee_angle = calculate_angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
            hip_angle = calculate_angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
            
            # Vertical travel (In MediaPipe, higher Y = closer to floor)
            curr_hip_y = lm["left_hip"][1] 
            curr_ankle_y = lm["left_ankle"][1]

            # 2. THE PRO-ML STATE MACHINE
            
            # STEP 1: RESET / STANDING
            if knee_angle > 165:
                if state == "ASCENDING" and max_depth_reached:
                    reps += 1
                    feedback = "Perfect Rep!"
                    color = (0, 255, 0)
                
                state = "STANDING"
                max_depth_reached = False
                hip_start_y = curr_hip_y # Set baseline for this specific rep
                if feedback != "Perfect Rep!":
                    feedback = "Ready"
                    color = (255, 255, 255)

            # STEP 2: SQUATTING (Going Down)
            elif state == "STANDING" and knee_angle < 130:
                # Posture Check: Is the chest up?
                if hip_angle > 100:
                    # Travel Check: Have hips moved down at least 10% relative to ankle?
                    vertical_travel = curr_hip_y - hip_start_y
                    if vertical_travel > 0.05: # Significant downward movement
                        state = "SQUATTING"
                        feedback = "Going Down..."
                else:
                    feedback = "POSTURE ERROR: Chest Up!"
                    color = (0, 0, 255)

            # STEP 3: BOTTOM REACHED
            elif state == "SQUATTING" and knee_angle < 105:
                # Depth Check: Ensure hip is not touching floor
                if abs(curr_hip_y - curr_ankle_y) > 0.15:
                    max_depth_reached = True
                    feedback = "Bottom Reached - Now Rise"
                    color = (0, 255, 255)
                else:
                    feedback = "ERROR: Don't sit on floor!"
                    state = "ERROR"

            # STEP 4: ASCENDING (Coming Up)
            elif (state == "SQUATTING" or state == "BOTTOM") and knee_angle > 135:
                if max_depth_reached:
                    state = "ASCENDING"
                    feedback = "Rising..."
                else:
                    feedback = "Not deep enough!"
                    state = "ERROR"

            # 3. DRAWING
            for conn in [(11,23),(23,25),(25,27),(11,12),(23,24)]:
                p1 = (int(landmarks[conn[0]].x * w), int(landmarks[conn[0]].y * h))
                p2 = (int(landmarks[conn[1]].x * w), int(landmarks[conn[1]].y * h))
                cv2.line(frame, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)

        # --- UI DISPLAY ---
        cv2.rectangle(frame, (0, 0), (500, 150), (20, 20, 20), -1)
        cv2.putText(frame, f"REPS: {reps}", (20, 50), 1, 3, (0, 255, 0), 3)
        cv2.putText(frame, f"STATE: {state}", (20, 95), 1, 1.2, (200, 200, 200), 2)
        cv2.putText(frame, f"{feedback}", (20, 135), 1, 1.5, color, 2)

        cv2.imshow("ML-Standard Squat Logic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pro_squat()