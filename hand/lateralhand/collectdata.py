import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# ==========================================
# MODEL PATH
# ==========================================
MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"

# ==========================================
# LOAD MODEL
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4, # Lowered slightly to reduce flickering
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# DATASET FILE
# ==========================================
dataset_file = "lateral_dual_hand_dataset.csv"

if not os.path.exists(dataset_file):
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")
    pd.DataFrame(columns=columns).to_csv(dataset_file, index=False)

# Hand Connections
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# CAMERA
# ==========================================
camera_index = 2 # Change to 0 for internal, 1 or 2 for external
cap = cv2.VideoCapture(camera_index)

# ==========================================
# FLICKER REDUCTION VARIABLES
# ==========================================
last_points = None
persistence_counter = 0
MAX_PERSISTENCE = 5 # Number of frames to keep lines if hand is lost

print("✅ Camera connected - Dual Lateral Data Collection")
print("L = Save LEFT LATERAL | R = Save RIGHT LATERAL | W = Save WRONG | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    current_row = None

    if result.hand_landmarks:
        persistence_counter = MAX_PERSISTENCE
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))
            
            current_row = row
            last_points = points
    else:
        if persistence_counter > 0:
            persistence_counter -= 1
        else:
            last_points = None
            current_row = None

    # DRAWING (Uses last_points to prevent flickering)
    if last_points:
        for connection in HAND_CONNECTIONS:
            cv2.line(frame, last_points[connection[0]], last_points[connection[1]], (255, 0, 0), 3)
        for pt in last_points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # TEXT OVERLAY
    cv2.putText(frame, "L = LEFT LATERAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "R = RIGHT LATERAL", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    cv2.putText(frame, "W = WRONG", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Data Collector - Dual Lateral", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('l'):
        if current_row:
            save_row = current_row.copy()
            save_row.append("Left")
            pd.DataFrame([save_row]).to_csv(dataset_file, mode='a', header=False, index=False)
            print("✅ Saved: Left")

    elif key == ord('r'):
        if current_row:
            save_row = current_row.copy()
            save_row.append("Right")
            pd.DataFrame([save_row]).to_csv(dataset_file, mode='a', header=False, index=False)
            print("✅ Saved: Right")

    elif key == ord('w'):
        if current_row:
            save_row = current_row.copy()
            save_row.append("Wrong")
            pd.DataFrame([save_row]).to_csv(dataset_file, mode='a', header=False, index=False)
            print("✅ Saved: Wrong")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()