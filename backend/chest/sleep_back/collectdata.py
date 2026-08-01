import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
CSV_FILE = "sleep_back_dataset.csv"

# --- INITIALIZE DETECTOR ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- CAMERA WARMUP ---
# Using CAP_DSHOW is mandatory on Windows to prevent the window from hanging
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Could not open camera. Try index 0 or check DroidCam.")
    exit()

# Force window to top
cv2.namedWindow("Physio AI Collector", cv2.WINDOW_NORMAL)
cv2.startWindowThread() 

print("🚀 Starting Stream... If window doesn't appear, check your taskbar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Empty Frame")
        break

    # Process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Non-blocking inference
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape

    if result.pose_landmarks:
        l = result.pose_landmarks[0]
        
        # Einstein Math for Stability
        mid_sh_y = (l[11].y + l[12].y) / 2
        mid_hp_y = (l[23].y + l[24].y) / 2
        torso_h = max(abs(mid_hp_y - mid_sh_y), 0.1)
        tilt = abs(l[11].y - l[12].y) / torso_h
        
        is_ready = tilt < 0.12
        color = (0, 255, 0) if is_ready else (0, 0, 255)

        # Draw Full Skeleton Lines
        pixel_pts = {i: (int((1 - l[i].x) * w), int(l[i].y * h)) for i in range(33)}
        connections = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (12, 14), (23, 25), (24, 26)]
        
        for start, end in connections:
            cv2.line(display_frame, pixel_pts[start], pixel_pts[end], color, 2)

        cv2.putText(display_frame, f"READY: {is_ready}", (20, 50), 1, 1.5, color, 2)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and is_ready:
            data = []
            for lm in l: data.extend([lm.x, lm.y, lm.z])
            pd.DataFrame([data + ["Correct_Sleep_Back"]]).to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            print("✔ Captured Correct")
        elif key == ord('w'):
            data = []
            for lm in l: data.extend([lm.x, lm.y, lm.z])
            pd.DataFrame([data + ["Wrong_Posture"]]).to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            print("✖ Captured Wrong")
        elif key == ord('q'): break
    else:
        cv2.putText(display_frame, "Detecting...", (20, 50), 1, 1, (0, 255, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.imshow("Physio AI Collector", display_frame)

cap.release()
cv2.destroyAllWindows()
detector.close()