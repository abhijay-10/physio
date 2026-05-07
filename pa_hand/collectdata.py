# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import csv
# import os

# # -------- Configuration --------
# MODEL_PATH = "hand_landmarker.task"
# CSV_PATH = "pa_hand_data.csv"
# CAMERA_INDEX = 1   # USB camera index

# if not os.path.exists(MODEL_PATH):
#     print(f"Error: {MODEL_PATH} not found.")
#     exit()

# # -------- MediaPipe Setup (TASKS ONLY) --------
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=1,
#     min_hand_detection_confidence=0.7,
#     min_hand_presence_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# detector = vision.HandLandmarker.create_from_options(options)

# # -------- CSV Setup --------
# file = open(CSV_PATH, mode="a", newline="")
# writer = csv.writer(file)

# if os.stat(CSV_PATH).st_size == 0:
#     writer.writerow(["w_z", "i_z", "m_z", "r_z", "p_z", "spread", "label"])

# # -------- Camera Setup --------
# cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     print(f"Error: Could not open USB camera at index {CAMERA_INDEX}")
#     exit()

# timestamp = 0

# print(f"--- PA HAND COLLECTOR ACTIVE (USB Camera Index {CAMERA_INDEX}) ---")
# print("R = Record RIGHT | W = Record WRONG | Q = Quit")

# # -------- Hand Connections (manual) --------
# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (5,9),(9,10),(10,11),(11,12),
#     (9,13),(13,14),(14,15),(15,16),
#     (13,17),(17,18),(18,19),(19,20),
#     (0,17)
# ]

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     key = cv2.waitKey(1) & 0xFF

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     result = detector.detect_for_video(mp_image, timestamp)
#     timestamp += 1

#     if result.hand_landmarks:
#         for landmarks in result.hand_landmarks:

#             # -------- Convert to pixel points --------
#             pts = []
#             for lm in landmarks:
#                 x = int(lm.x * w)
#                 y = int(lm.y * h)
#                 pts.append((x, y))

#             # -------- Draw connections --------
#             for c in HAND_CONNECTIONS:
#                 cv2.line(frame, pts[c[0]], pts[c[1]], (0, 255, 255), 2)

#             # -------- Draw points --------
#             for p in pts:
#                 cv2.circle(frame, p, 4, (0, 0, 255), -1)

#             # -------- Data Calculation --------
#             hand_spread = abs(landmarks[2].x - landmarks[17].x)

#             data_row = [
#                 landmarks[0].z,
#                 landmarks[8].z,
#                 landmarks[12].z,
#                 landmarks[16].z,
#                 landmarks[20].z,
#                 hand_spread
#             ]

#             # -------- Save RIGHT --------
#             if key == ord('r'):
#                 writer.writerow(data_row + [1])
#                 file.flush()
#                 print("Logged: RIGHT")

#                 cv2.rectangle(frame, (0, h - 50), (w, h), (0, 255, 0), -1)
#                 cv2.putText(frame, "SAVED: RIGHT",
#                             (w // 2 - 90, h - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (0, 0, 0), 2)

#             # -------- Save WRONG --------
#             elif key == ord('w'):
#                 writer.writerow(data_row + [0])
#                 file.flush()
#                 print("Logged: WRONG")

#                 cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
#                 cv2.putText(frame, "SAVED: WRONG",
#                             (w // 2 - 100, h - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (255, 255, 255), 2)

#     # -------- UI --------
#     cv2.putText(frame, "PA HAND POSITIONING GUIDE",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255), 2)

#     cv2.imshow("X-Ray Trainer - USB Camera", frame)

#     if key == ord('q'):
#         break

# # -------- Cleanup --------
# cap.release()
# file.close()
# cv2.destroyAllWindows()pa


# 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import joblib
import time

# -------- CONFIG --------
MODEL_PATH = "hand_landmarker.task"
CSV_PATH = "oblique_data.csv"
ML_MODEL_PATH = "oblique_model.pkl"
CAMERA_INDEX = 1  # change if needed

# -------- LOAD ML MODEL --------
if os.path.exists(ML_MODEL_PATH):
    model = joblib.load(ML_MODEL_PATH)
    print("✅ ML Model Loaded")
else:
    model = None
    print("⚠️ No trained model found")

# -------- CSV SETUP --------
file_exists = os.path.exists(CSV_PATH)
file = open(CSV_PATH, mode="a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(["w_z","i_z","m_z","r_z","p_z","spread","direction","label"])

# -------- MEDIAPIPE --------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -------- HAND CONNECTIONS --------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# -------- CAMERA --------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("\n🎯 CONTROLS: R=RIGHT | L=LEFT | W=WRONG | Q=QUIT\n")

timestamp = 0

# -------- LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    key = cv2.waitKey(1) & 0xFF

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:

            # -------- DRAW --------
            for c in HAND_CONNECTIONS:
                x1 = int(landmarks[c[0]].x * w)
                y1 = int(landmarks[c[0]].y * h)
                x2 = int(landmarks[c[1]].x * w)
                y2 = int(landmarks[c[1]].y * h)
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 2)

            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x,y), 4, (0,0,255), -1)

            # -------- FEATURES --------
            spread = abs(landmarks[2].x - landmarks[17].x)
            direction = landmarks[5].x - landmarks[17].x

            data_row = [
                landmarks[0].z,
                landmarks[8].z,
                landmarks[12].z,
                landmarks[16].z,
                landmarks[20].z,
                spread,
                direction
            ]

            # -------- SAVE DATA --------
            if key == ord('r'):
                writer.writerow(data_row + ["right"])
                print("✅ Saved: RIGHT")

            elif key == ord('l'):
                writer.writerow(data_row + ["left"])
                print("✅ Saved: LEFT")

            elif key == ord('w'):
                writer.writerow(data_row + ["wrong"])
                print("❌ Saved: WRONG")

            # -------- PREDICTION --------
            if model is not None:
                pred = model.predict([data_row])[0]

                if pred == 1:
                    label = "RIGHT"
                    color = (0,255,0)
                elif pred == 2:
                    label = "LEFT"
                    color = (255,0,0)
                else:
                    label = "WRONG"
                    color = (0,0,255)

                cv2.putText(frame, label, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # -------- UI --------
    cv2.putText(frame, "R=RIGHT L=LEFT W=WRONG Q=QUIT",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)

    cv2.imshow("PA Hand AI System", frame)

    if key == ord('q'):
        break

# -------- CLEANUP --------
cap.release()
file.close()
cv2.destroyAllWindows()