import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION & DATASET TARGET SETUP
# ==========================================
CSV_FILE = "foot_pose_dataset.csv"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")

HEADER = [
    "ankle_x", "ankle_y", "ankle_z", "ankle_vis",
    "heel_x", "heel_y", "heel_z", "heel_vis",
    "toe_x", "toe_y", "toe_z", "toe_vis",
    "label"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(HEADER)

# Open External Webcam Port Index 2 
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# ANTI-FLICKER TEMPORAL HISTORICAL SMOOTHER
# ==========================================
class CollectionStabilizer:
    def __init__(self, window_size=8):
        self.ax, self.ay, self.az, self.av = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)
        self.hx, self.hy, self.hz, self.hv = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)
        self.tx, self.ty, self.tz, self.tv = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)

    def push_and_smooth(self, ankle, heel, toe):
        self.ax.append(ankle.x); self.ay.append(ankle.y); self.az.append(ankle.z); self.av.append(ankle.visibility)
        self.hx.append(heel.x); self.hy.append(heel.y); self.hz.append(heel.z); self.hv.append(heel.visibility)
        self.tx.append(toe.x); self.ty.append(toe.y); self.tz.append(toe.z); self.tv.append(toe.visibility)
        
        return [
            np.mean(self.ax), np.mean(self.ay), np.mean(self.az), np.mean(self.av),
            np.mean(self.hx), np.mean(self.hy), np.mean(self.hz), np.mean(self.hv),
            np.mean(self.tx), np.mean(self.ty), np.mean(self.tz), np.mean(self.tv)
        ]

stabilizer = CollectionStabilizer(window_size=8)

# ==========================================
# INITIALIZE NEW VISION TASK OBJECT
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.10, # Lowered threshold to force detection through flicker
    min_tracking_confidence=0.20
)
detector = vision.PoseLandmarker.create_from_options(options)

# Recording session state machine parameters
is_recording = False
record_label = -1
record_end_time = 0
sample_counter = 0

print("\n⚡ AXORIS Physio Anti-Flicker Data Collector Subsystem Online...")
print("👉 Press [1] to burst record 3 seconds of RIGHT POSTURE")
print("👉 Press [0] to burst record 3 seconds of WRONG POSTURE")
print("👉 Press [Q] to Save & Close safely\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        display_frame = cv2.flip(frame, 1)
        h, w, _ = display_frame.shape

        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_wrapper = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        result = detector.detect_for_video(mp_image_wrapper, int(time.time() * 1000))

        # Non-blocking keystroke check (Reads inputs completely outside tracking state)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1') and not is_recording:
            is_recording = True
            record_label = 1
            record_end_time = time.time() + 3.0 # Set a 3-second continuous recording buffer
            print("🚀 Started burst collection for Class 1 (RIGHT)... Hold position still.")
        elif key == ord('0') and not is_recording:
            is_recording = True
            record_label = 0
            record_end_time = time.time() + 3.0
            print("🚀 Started burst collection for Class 0 (WRONG)... Hold position still.")
        elif key == ord('q') or key == ord('Q'):
            break

        # Check if the active recording window is running
        if is_recording:
            if time.time() > record_end_time:
                is_recording = False
                print(f"✅ Burst Finished! Total dataset sample rows logged: {sample_counter}")
            else:
                time_left = record_end_time - time.time()
                cv2.putText(display_frame, f"RECORDING CLASS {record_label} ({time_left:.1f}s)", (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

            # Choose active leg profile via visibility check balances
            left_score = landmarks[27].visibility + landmarks[29].visibility + landmarks[31].visibility
            right_score = landmarks[24].visibility + landmarks[26].visibility + landmarks[28].visibility

            if left_score > right_score:
                ankle_lm, heel_lm, toe_lm = landmarks[27], landmarks[29], landmarks[31]
                lbl = "LEFT FOOT PROFILE"
            else:
                ankle_lm, heel_lm, toe_lm = landmarks[28], landmarks[30], landmarks[32]
                lbl = "RIGHT FOOT PROFILE"

            # Smooth data points over time to clear the flicker
            smoothed_features = stabilizer.push_and_smooth(ankle_lm, heel_lm, toe_lm)

            # Map coordinates for UI tracking circles
            a_pt = (int((1 - smoothed_features[1]) * w), int(smoothed_features[1] * h))
            h_pt = (int((1 - smoothed_features[5]) * w), int(smoothed_features[5] * h))
            t_pt = (int((1 - smoothed_features[9]) * w), int(smoothed_features[9] * h))

            # UI skeletal overlays
            cv2.line(display_frame, h_pt, a_pt, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(display_frame, a_pt, t_pt, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display_frame, a_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display_frame, h_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(display_frame, t_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(display_frame, f"{lbl} (Stabilized)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If recording mode is triggered, write the smoothed dataset row automatically
            if is_recording:
                with open(CSV_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow(smoothed_features + [record_label])
                sample_counter += 1
        else:
            cv2.putText(display_frame, "SEARCHING FOR FOOT FIELDS...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Axoris Ultra-Stable Data Collector Suite", display_frame)

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nDataset generation closed cleanly. Total records in file: {sample_counter} rows.")