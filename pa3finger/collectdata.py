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
dataset_file = "pa_finger_dataset.csv"
camera_index = 2  # Set to 0 for internal, 1 or 2 for external

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

# ==========================================
# FILTERED CONNECTIONS (Index, Middle, Ring)
# ==========================================
# Landmarks: 0=Wrist, 5-8=Forefinger, 9-12=Middle, 13-16=Ring
PA_FINGER_CONNECTIONS = [
    (0,5), (5,6), (6,7), (7,8),       # Forefinger (Index)
    (0,9), (9,10), (10,11), (11,12),    # Middle Finger
    (0,13), (13,14), (14,15), (15,16)   # Ring Finger
]
# Only draw these points on screen
ALLOWED_POINTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

# Persistence Variables to stop flickering
last_points = None
last_row = None
persistence_counter = 0
MAX_PERSISTENCE = 10 

cap = cv2.VideoCapture(camera_index)

print("✅ PA Finger Collector Started (Index, Middle, Ring Only)")
print("S = Save PA FINGER | W = Save WRONG | Q = Quit")

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

    # DRAWING (Filtered for Fore, Middle, Ring)
    if last_points:
        for conn in PA_FINGER_CONNECTIONS:
            cv2.line(frame, last_points[conn[0]], last_points[conn[1]], (0, 255, 0), 3)
        
        for i in ALLOWED_POINTS:
            cv2.circle(frame, last_points[i], 5, (255, 255, 255), -1)

    # UI TEXT
    cv2.putText(frame, "S = SAVE PA FINGER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "W = WRONG | Q = QUIT", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("PA Finger Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and last_row:
        save_data = last_row.copy()
        save_data.append("PA Finger")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: PA FINGER (Index/Middle/Ring)")

    elif key == ord('w') and last_row:
        save_data = last_row.copy()
        save_data.append("Wrong")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: WRONG")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()