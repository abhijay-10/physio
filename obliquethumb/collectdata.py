import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "d:\\physio\\obliquehand\\hand_landmarker.task"
dataset_file = "oblique_thumb_dataset.csv"
camera_index = 2  # Use 0 for internal, 1 or 2 for external

# ==========================================
# LOAD MEDIAPIPE MODEL
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# INITIALIZE CSV
# ==========================================
if not os.path.exists(dataset_file):
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")
    pd.DataFrame(columns=columns).to_csv(dataset_file, index=False)

# Thumb specific landmarks: 0 (Wrist), 1, 2, 3, 4
THUMB_INDICES = [0, 1, 2, 3, 4]
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# ==========================================
# ANTI-FLICKER VARIABLES
# ==========================================
last_points = None
last_row = None
persistence_counter = 0
MAX_PERSISTENCE = 10 

cap = cv2.VideoCapture(camera_index)

print("✅ Oblique Thumb Collector Started")
print("T = Save OBLIQUE THUMB | W = Save WRONG | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        persistence_counter = MAX_PERSISTENCE
        for hand_landmarks in result.hand_landmarks:
            row = []
            points = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])
                points.append((int(lm.x * w), int(lm.y * h)))
            
            last_row = row
            last_points = points
    else:
        if persistence_counter > 0:
            persistence_counter -= 1
        else:
            last_points = None
            last_row = None

    # DRAWING
    if last_points:
        for conn in HAND_CONNECTIONS:
            # Use Green for thumb connections, Blue for the rest
            color = (0, 255, 0) if conn[0] in THUMB_INDICES and conn[1] in THUMB_INDICES else (255, 0, 0)
            cv2.line(frame, last_points[conn[0]], last_points[conn[1]], color, 2)
        
        for i, pt in enumerate(last_points):
            # Highlight thumb tip
            color = (0, 255, 255) if i == 4 else (255, 255, 255)
            cv2.circle(frame, pt, 4, color, -1)

    # UI TEXT
    cv2.putText(frame, "T = SAVE OBLIQUE THUMB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "W = WRONG | Q = QUIT", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Oblique Thumb Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t') and last_row:
        save_data = last_row.copy()
        save_data.append("Oblique Thumb")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: OBLIQUE THUMB")

    elif key == ord('w') and last_row:
        save_data = last_row.copy()
        save_data.append("Wrong")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: WRONG")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()