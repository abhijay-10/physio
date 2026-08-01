import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/") 
CSV_FILE = "chest_ap_stable_dataset.csv"

# --- 2. INITIALIZE DETECTOR (Stricter Settings to stop flicker) ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.7, # Strict detection
    min_tracking_confidence=0.8        # Very strict tracking to kill jitter
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- 3. CAMERA SETUP ---
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW) 
cv2.namedWindow("Stable Radiology System", cv2.WINDOW_NORMAL)

count_r, count_w = 0, 0
# Torso and Neck/Head markers for the semi-upright posture
CHEST_SKELETON = [(11, 12), (23, 24), (11, 23), (12, 24), (11, 7), (12, 8)]

print("📡 System Online. Hold R for Correct, W for Wrong.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Clinical view
    display_frame = frame.copy()
    h, w, _ = display_frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    label = None
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'): label, count_r = "Correct_AP", count_r + 1
    elif key == ord('w'): label, count_w = "Wrong_AP", count_w + 1

    if result.pose_landmarks:
        l = result.pose_landmarks[0]
        px = {i: (int(l[i].x * w), int(l[i].y * h)) for i in range(33)}
        
        # --- DRAWING THE CHEST BOX ---
        # We only draw if the points are highly visible to ensure "Correctness"
        for s, e in CHEST_SKELETON:
            if l[s].visibility > 0.7 and l[e].visibility > 0.7:
                cv2.line(display_frame, px[s], px[e], (0, 255, 0), 3)
        
        # Highlight the main "Anchor" points
        for i in [11, 12, 23, 24]:
            if l[i].visibility > 0.7:
                cv2.circle(display_frame, px[i], 7, (255, 255, 255), -1)

        # --- DATA SAVE LOGIC ---
        if label:
            # We save all coordinates for the Random Forest
            features = []
            for lm in l: features.extend([lm.x, lm.y, lm.z])
            features.append(label)
            
            pd.DataFrame([features]).to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

    # --- UI HUD ---
    cv2.rectangle(display_frame, (10, 10), (280, 95), (40, 40, 40), -1)
    cv2.putText(display_frame, f"Correct R: {count_r}", (20, 40), 1, 1.2, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Wrong W: {count_w}", (20, 75), 1, 1.2, (0, 0, 255), 2)
    
    if label:
        cv2.putText(display_frame, "RECORDING...", (w-180, 40), 1, 1.2, (0, 255, 255), 2)

    cv2.imshow("Stable Radiology System", display_frame)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
detector.close()