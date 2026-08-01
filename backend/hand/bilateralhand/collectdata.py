import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"
dataset_file = "bilateral_pa_dataset.csv"
camera_index = 2  # Set to 0 for internal, 1 or 2 for external

# ==========================================
# LOAD MEDIAPIPE (Configured for 2 Hands)
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # CRITICAL: Detect both hands
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# INITIALIZE CSV (126 coordinates for 2 hands)
# ==========================================
if not os.path.exists(dataset_file):
    columns = []
    # 2 hands * 21 landmarks * 3 coordinates (x,y,z) = 126 columns
    for hand_num in range(2):
        for i in range(21):
            columns.extend([f"h{hand_num}_x{i}", f"h{hand_num}_y{i}", f"h{hand_num}_z{i}"])
    columns.append("label")
    pd.DataFrame(columns=columns).to_csv(dataset_file, index=False)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# Anti-Flicker Variables
last_full_row = None
last_hands_points = []
persistence_counter = 0
MAX_PERSISTENCE = 10 

cap = cv2.VideoCapture(camera_index)

print("✅ Bilateral PA Collector Started")
print("S = Save BILATERAL PA (Both hands must be visible) | W = Save WRONG | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    current_frame_row = []
    current_frame_points = []

    # Check if exactly 2 hands are detected for a valid Bilateral capture
    if result.hand_landmarks and len(result.hand_landmarks) == 2:
        persistence_counter = MAX_PERSISTENCE
        
        # Sort hands by x-coordinate to ensure h0 is always the left-most hand in frame
        sorted_hands = sorted(result.hand_landmarks, key=lambda x: x[0].x)
        
        for hand_landmarks in sorted_hands:
            hand_pts = []
            for lm in hand_landmarks:
                current_frame_row.extend([lm.x, lm.y, lm.z])
                hand_pts.append((int(lm.x * w), int(lm.y * h)))
            current_frame_points.append(hand_pts)
        
        last_full_row = current_frame_row
        last_hands_points = current_frame_points
        status_color = (0, 255, 0) # Green: Ready
        status_txt = "READY: 2 HANDS DETECTED"
    else:
        if persistence_counter > 0:
            persistence_counter -= 1
            status_color = (0, 255, 255) # Yellow: Holding
            status_txt = "HOLDING STABILITY..."
        else:
            last_hands_points = []
            last_full_row = None
            status_color = (0, 0, 255) # Red: Error
            status_txt = "ERROR: NEED BOTH HANDS"

    # DRAWING
    for hand_pts in last_hands_points:
        for conn in HAND_CONNECTIONS:
            cv2.line(frame, hand_pts[conn[0]], hand_pts[conn[1]], status_color, 2)
        for pt in hand_pts:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)

    cv2.putText(frame, status_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "S = SAVE | W = WRONG", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Bilateral PA Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and last_full_row:
        save_data = last_full_row.copy()
        save_data.append("Bilateral PA")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: BILATERAL PA")

    elif key == ord('w') and last_full_row:
        save_data = last_full_row.copy()
        save_data.append("Wrong")
        pd.DataFrame([save_data]).to_csv(dataset_file, mode='a', header=False, index=False)
        print("✅ Saved: WRONG")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()