import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")
OUTPUT_CSV = "sleep_front_dataset.csv"
CAMERA_INDEX = 2  # DroidCam index

# ==========================================
# INITIALIZE
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Torso connections: Shoulders (11,12) to Hips (23,24)
TORSO_SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24)]

dataset = []
print("--- Sleep Front (Supine) Pose Collection ---")
print("HOLD [r]: Correct Supine Pose | HOLD [w]: Wrong (Tilted/Crooked) | [q]: Save & Quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mirroring for the user, but we collect raw data
    display_frame = cv2.flip(frame, 1)
    h, w, _ = display_frame.shape
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    ms_timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, ms_timestamp)

    # Keyboard logic
    key = cv2.waitKey(1) & 0xFF
    current_label = None
    hud_color = (200, 200, 200) 
    status_msg = "IDLE - LAY FLAT ON BACK"

    if key == ord('r'):
        current_label, status_msg, hud_color = "Correct_Sleep_Front", "RECORDING SUPINE POSE...", (0, 255, 0)
    elif key == ord('w'):
        current_label, status_msg, hud_color = "Wrong_Posture", "RECORDING WRONG...", (0, 0, 255)
    elif key == ord('q'):
        break

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        # Mirror coordinates for display skeleton
        pts = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in landmarks]

        # Draw Torso Skeleton
        for conn in TORSO_SKELETON:
            cv2.line(display_frame, pts[conn[0]], pts[conn[1]], hud_color, 3)
        for idx in [11, 12, 23, 24]:
            cv2.circle(display_frame, pts[idx], 8, (255, 255, 255), -1)

        # Record raw (unflipped) data
        if current_label:
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])
            row.append(current_label)
            dataset.append(row)

    # UI Overlay
    cv2.rectangle(display_frame, (0, h-80), (w, h), (15, 15, 15), -1)
    cv2.putText(display_frame, f"{status_msg} | Samples: {len(dataset)}", (50, h-30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, hud_color, 2)
    
    cv2.imshow("Physio AI - Sleep Front Collection", display_frame)

cap.release()
cv2.destroyAllWindows()

# Save logic
if dataset:
    cols = [f'x{i}' for i in range(33)] + [f'y{i}' for i in range(33)] + [f'z{i}' for i in range(33)] + ['target']
    header_needed = not os.path.exists(OUTPUT_CSV)
    pd.DataFrame(dataset, columns=cols).to_csv(OUTPUT_CSV, mode='a', index=False, header=header_needed)
    print(f"✅ Saved {len(dataset)} samples to {OUTPUT_CSV}")