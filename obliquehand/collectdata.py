import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# ==========================================
# MODEL PATH
# ==========================================
MODEL_PATH = "hand_landmarker.task"

# ==========================================
# LOAD MODEL
# ==========================================
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)

# ==========================================
# DATASET FILE
# ==========================================
dataset_file = "hand_dataset.csv"

if not os.path.exists(dataset_file):

    columns = []

    for i in range(21):

        columns.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    columns.append("label")

    pd.DataFrame(columns=columns).to_csv(
        dataset_file,
        index=False
    )

# ==========================================
# HAND CONNECTIONS
# ==========================================
HAND_CONNECTIONS = [

    (0,1), (1,2), (2,3), (3,4),

    (0,5), (5,6), (6,7), (7,8),

    (0,9), (9,10), (10,11), (11,12),

    (0,13), (13,14), (14,15), (15,16),

    (0,17), (17,18), (18,19), (19,20),

    (5,9), (9,13), (13,17)
]

# ==========================================
# CAMERA
# ==========================================
# CHANGE CAMERA INDEX IF NEEDED
camera_index = 2

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():

    print("❌ Camera not opening")
    exit()

print("✅ Camera connected")

print("\n===================================")
print("HAND DATA COLLECTION")
print("===================================")
print("L = Save LEFT")
print("R = Save RIGHT")
print("W = Save WRONG")
print("Q = Quit")
print("===================================\n")

# ==========================================
# MAIN LOOP
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    current_row = None

    # ======================================
    # HAND DETECTION
    # ======================================
    if result.hand_landmarks:

        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            row = []
            points = []

            # --------------------------------
            # LANDMARKS
            # --------------------------------
            for lm in hand_landmarks:

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                points.append((cx, cy))

            current_row = row

            # --------------------------------
            # DRAW SKELETON
            # --------------------------------
            for connection in HAND_CONNECTIONS:

                start_idx, end_idx = connection

                x1, y1 = points[start_idx]
                x2, y2 = points[end_idx]

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255,0,0),
                    3
                )

            # --------------------------------
            # DRAW LANDMARKS
            # --------------------------------
            for point in points:

                cv2.circle(
                    frame,
                    point,
                    6,
                    (0,255,0),
                    -1
                )

    # ======================================
    # SCREEN TEXT
    # ======================================
    cv2.putText(
        frame,
        "L = LEFT",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "R = RIGHT",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "W = WRONG",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        "Q = QUIT",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    # ======================================
    # SHOW WINDOW
    # ======================================
    cv2.imshow(
        "Hand Dataset Collector",
        frame
    )

    # ======================================
    # KEYS
    # ======================================
    key = cv2.waitKey(1) & 0xFF

    # SAVE LEFT
    if key == ord('l'):

        if current_row is not None:

            save_row = current_row.copy()

            save_row.append("Left")

            pd.DataFrame([save_row]).to_csv(
                dataset_file,
                mode='a',
                header=False,
                index=False
            )

            print("✅ LEFT saved")

    # SAVE RIGHT
    elif key == ord('r'):

        if current_row is not None:

            save_row = current_row.copy()

            save_row.append("Right")

            pd.DataFrame([save_row]).to_csv(
                dataset_file,
                mode='a',
                header=False,
                index=False
            )

            print("✅ RIGHT saved")

    # SAVE WRONG
    elif key == ord('w'):

        if current_row is not None:

            save_row = current_row.copy()

            save_row.append("Wrong")

            pd.DataFrame([save_row]).to_csv(
                dataset_file,
                mode='a',
                header=False,
                index=False
            )

            print("✅ WRONG saved")

    # QUIT
    elif key == ord('q'):

        break

# ==========================================
# CLOSE
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("Program closed")