# =========================================================
# STABLE FOOT POSTURE DATASET COLLECTOR
# LOWER BODY + FOOT TRACKING
#
# KEYS:
# r -> RIGHT POSTURE
# w -> WRONG POSTURE
# q -> SAVE & EXIT
# =========================================================

import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = r"D:\physio\pose_landmarker_full.task"

OUTPUT_CSV = "foot_posture_dataset.csv"

CAMERA_INDEX = 2

# =========================================================
# MEDIAPIPE TASK
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.8,

    min_pose_presence_confidence=0.8,

    min_tracking_confidence=0.8
)

detector = vision.PoseLandmarker.create_from_options(options)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================================================
# LOWER BODY CONNECTIONS
# =========================================================

BODY_CONNECTIONS = [

    # LEFT LEG
    (23, 25),
    (25, 27),

    # RIGHT LEG
    (24, 26),
    (26, 28),

    # LEFT FOOT
    (27, 29),
    (29, 31),

    # RIGHT FOOT
    (28, 30),
    (30, 32),

    # ANKLE CONNECTION
    (27, 28)
]

# =========================================================
# FOOT LANDMARK IDS
# =========================================================

FOOT_IDS = [

    25,  # LEFT KNEE
    26,  # RIGHT KNEE

    27,  # LEFT ANKLE
    28,  # RIGHT ANKLE

    29,  # LEFT HEEL
    30,  # RIGHT HEEL

    31,  # LEFT TOE
    32   # RIGHT TOE
]

# =========================================================
# DATASET
# =========================================================

dataset = []

right_count = 0
wrong_count = 0

# =========================================================
# SMOOTHING
# =========================================================

previous_points = None

SMOOTHING = 0.75

# =========================================================
# FPS
# =========================================================

prev_time = time.time()

# =========================================================
# STABLE TIMESTAMP
# =========================================================

frame_timestamp_ms = 0

# =========================================================
# ANGLE FUNCTION
# =========================================================

def calculate_angle(a, b, c):

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cosine = dot / (mag_ba * mag_bc)

    cosine = max(-1, min(1, cosine))

    angle = math.degrees(math.acos(cosine))

    return angle

# =========================================================
# INFO
# =========================================================

print("\n========== FOOT POSTURE COLLECTOR ==========")

print("r -> RIGHT POSTURE")
print("w -> WRONG POSTURE")
print("q -> SAVE & EXIT")

# =========================================================
# LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # =====================================================
    # RGB
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # =====================================================
    # STABLE TIMESTAMP
    # =====================================================

    frame_timestamp_ms += 33

    # =====================================================
    # DETECTION
    # =====================================================

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    # =====================================================
    # LANDMARKS
    # =====================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        current_points = []

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            current_points.append((x, y))

        # =================================================
        # SMOOTH LANDMARKS
        # =================================================

        if previous_points is not None:

            smoothed_points = []

            for curr, prev in zip(
                current_points,
                previous_points
            ):

                smooth_x = int(
                    prev[0] * SMOOTHING +
                    curr[0] * (1 - SMOOTHING)
                )

                smooth_y = int(
                    prev[1] * SMOOTHING +
                    curr[1] * (1 - SMOOTHING)
                )

                smoothed_points.append(
                    (smooth_x, smooth_y)
                )

            pixel_points = smoothed_points

        else:

            pixel_points = current_points

        previous_points = pixel_points

        # =================================================
        # DRAW CONNECTIONS
        # =================================================

        for conn in BODY_CONNECTIONS:

            start = pixel_points[conn[0]]
            end = pixel_points[conn[1]]

            cv2.line(
                frame,
                start,
                end,
                (0, 255, 0),
                3
            )

        # =================================================
        # DRAW FOOT POINTS
        # =================================================

        for idx in FOOT_IDS:

            x, y = pixel_points[idx]

            cv2.circle(
                frame,
                (x, y),
                7,
                (0, 0, 255),
                -1
            )

        # =================================================
        # LABELS
        # =================================================

        labels = {

            25: "L_KNEE",
            26: "R_KNEE",

            27: "L_ANKLE",
            28: "R_ANKLE",

            29: "L_HEEL",
            30: "R_HEEL",

            31: "L_TOE",
            32: "R_TOE"
        }

        for idx, text in labels.items():

            x, y = pixel_points[idx]

            cv2.putText(
                frame,
                text,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        # =================================================
        # ANGLES
        # =================================================

        left_knee_angle = calculate_angle(

            pixel_points[23],
            pixel_points[25],
            pixel_points[27]
        )

        right_knee_angle = calculate_angle(

            pixel_points[24],
            pixel_points[26],
            pixel_points[28]
        )

        left_foot_angle = calculate_angle(

            pixel_points[29],
            pixel_points[27],
            pixel_points[31]
        )

        right_foot_angle = calculate_angle(

            pixel_points[30],
            pixel_points[28],
            pixel_points[32]
        )

        # =================================================
        # DISPLAY ANGLES
        # =================================================

        cv2.putText(
            frame,
            f"L Foot Angle: {int(left_foot_angle)}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"R Foot Angle: {int(right_foot_angle)}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # =================================================
        # KEYS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        label = None

        if key == ord('r'):

            label = "RIGHT_POSTURE"

            right_count += 1

        elif key == ord('w'):

            label = "WRONG_POSTURE"

            wrong_count += 1

        # =================================================
        # SAVE DATA
        # =================================================

        if label is not None:

            row = []

            # SAVE ONLY LOWER BODY LANDMARKS
            for idx in FOOT_IDS:

                lm = landmarks[idx]

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            # SAVE ANGLES
            row.extend([

                left_knee_angle,
                right_knee_angle,

                left_foot_angle,
                right_foot_angle
            ])

            row.append(label)

            dataset.append(row)

            cv2.putText(
                frame,
                f"{label} SAVED",
                (220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # =================================================
        # EXIT
        # =================================================

        if key == ord('q'):
            break

    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # =====================================================
    # HUD
    # =====================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 140),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"RIGHT: {right_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"WRONG: {wrong_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "FOOT POSTURE COLLECTOR",
        frame
    )

# =========================================================
# RELEASE
# =========================================================

cap.release()

cv2.destroyAllWindows()

# =========================================================
# SAVE CSV
# =========================================================

if len(dataset) > 0:

    columns = []

    for i in FOOT_IDS:

        columns.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    # ANGLE FEATURES
    columns.extend([

        "left_knee_angle",
        "right_knee_angle",

        "left_foot_angle",
        "right_foot_angle"
    ])

    columns.append("target")

    df = pd.DataFrame(
        dataset,
        columns=columns
    )

    if not os.path.isfile(OUTPUT_CSV):

        df.to_csv(
            OUTPUT_CSV,
            index=False
        )

    else:

        df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=False,
            index=False
        )

    print(f"\n✅ Saved {len(dataset)} samples")

else:

    print("\nNo data collected")