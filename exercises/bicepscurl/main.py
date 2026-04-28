# import cv2
# import numpy as np
# import time
# import joblib

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # -------- Load ML Model --------
# model = joblib.load("exercise_model.pkl")

# # -------- Angle Function --------
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
#               np.arctan2(a[1]-b[1], a[0]-b[0])

#     angle = abs(radians * 180.0 / np.pi)

#     if angle > 180:
#         angle = 360 - angle

#     return angle

# # -------- Camera --------
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# # -------- MediaPipe --------
# base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO
# )

# detector = vision.PoseLandmarker.create_from_options(options)

# # -------- Variables --------
# counter = 0
# stage = None
# start_time = 0
# timestamp = 0

# feedback = ""
# last_rep_time = 0
# cooldown = 1.0

# # -------- LOOP --------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     result = detector.detect_for_video(mp_image, timestamp)
#     timestamp += 1

#     try:
#         if result.pose_landmarks:
#             lm = result.pose_landmarks[0]

#             # -------- BOTH ARM DETECTION --------
#             l_vis = min(lm[11].visibility, lm[13].visibility, lm[15].visibility)
#             r_vis = min(lm[12].visibility, lm[14].visibility, lm[16].visibility)

#             if r_vis > l_vis:
#                 shoulder = [lm[12].x, lm[12].y]
#                 elbow = [lm[14].x, lm[14].y]
#                 wrist = [lm[16].x, lm[16].y]
#                 arm = "RIGHT"
#             else:
#                 shoulder = [lm[11].x, lm[11].y]
#                 elbow = [lm[13].x, lm[13].y]
#                 wrist = [lm[15].x, lm[15].y]
#                 arm = "LEFT"

#             # -------- Angle --------
#             angle = calculate_angle(shoulder, elbow, wrist)

#             # -------- Rep Logic --------
#             if angle > 160:
#                 stage = "down"
#                 start_time = time.time()

#             if angle < 40 and stage == "down":
#                 now = time.time()

#                 if now - last_rep_time > cooldown:
#                     stage = "up"
#                     last_rep_time = now

#                     rep_time = now - start_time
#                     shoulder_move = abs(shoulder[0] - elbow[0])

#                     features = [[angle, shoulder_move, rep_time]]
#                     pred = model.predict(features)[0]

#                     if pred == 1:
#                         counter += 1
#                         feedback = "Correct ✅"
#                     else:
#                         feedback = "Wrong ❌"

#             # -------- Draw Skeleton --------
#             pts = [
#                 (int(shoulder[0]*w), int(shoulder[1]*h)),
#                 (int(elbow[0]*w), int(elbow[1]*h)),
#                 (int(wrist[0]*w), int(wrist[1]*h))
#             ]

#             cv2.line(frame, pts[0], pts[1], (0,255,255), 3)
#             cv2.line(frame, pts[1], pts[2], (0,255,255), 3)
#             for p in pts:
#                 cv2.circle(frame, p, 6, (0,0,255), -1)

#             # -------- Display --------
#             cv2.putText(frame, f"Angle: {int(angle)}", pts[1],
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#             cv2.putText(frame, f"Arm: {arm}",
#                         (10,60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#     except:
#         pass

#     # -------- UI --------
#     cv2.rectangle(frame, (0,0), (640,50), (0,0,0), -1)

#     cv2.putText(frame, f"Reps: {counter}",
#                 (10,30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.putText(frame, feedback,
#                 (200,30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

#     cv2.imshow("AI Physio Trainer", frame)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# # -------- Cleanup --------
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import joblib

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- Load Model --------
model = joblib.load("exercise_model.pkl")

# -------- Angle Function --------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180.0 / np.pi)
    return 360-angle if angle > 180 else angle


# -------- Progress Bar --------
def draw_progress_bar(frame, angle):
    bar_x, bar_y = 50, 420
    bar_w, bar_h = 540, 20

    progress = np.interp(angle, (40, 160), (0, bar_w))

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (50,50,50), -1)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (int(bar_x + progress), bar_y + bar_h),
                  (0,255,0), -1)

    cv2.putText(frame, "Range of Motion",
                (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


# -------- Camera --------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# -------- MediaPipe --------
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

# -------- Variables --------
counter = 0
stage = None
start_time = 0
timestamp = 0

feedback = ""
last_rep_time = 0
cooldown = 1.0


# -------- MAIN LOOP --------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    try:
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # -------- BOTH ARM DETECTION --------
            l_vis = min(lm[11].visibility, lm[13].visibility, lm[15].visibility)
            r_vis = min(lm[12].visibility, lm[14].visibility, lm[16].visibility)

            if r_vis > l_vis:
                shoulder = [lm[12].x, lm[12].y]
                elbow = [lm[14].x, lm[14].y]
                wrist = [lm[16].x, lm[16].y]
                arm = "RIGHT"
            else:
                shoulder = [lm[11].x, lm[11].y]
                elbow = [lm[13].x, lm[13].y]
                wrist = [lm[15].x, lm[15].y]
                arm = "LEFT"

            # -------- Angle --------
            angle = calculate_angle(shoulder, elbow, wrist)

            # -------- REP LOGIC --------
            if angle > 160:
                stage = "down"
                start_time = time.time()

            if angle < 40 and stage == "down":
                now = time.time()

                if now - last_rep_time > cooldown:
                    stage = "up"
                    last_rep_time = now

                    rep_time = now - start_time
                    shoulder_move = abs(shoulder[0] - elbow[0])

                    features = [[angle, shoulder_move, rep_time]]
                    pred = model.predict(features)[0]

                    if pred == 1:
                        counter += 1
                        feedback = "Correct ✅"
                    else:
                        feedback = "Wrong ❌"

            # -------- Beautiful Skeleton --------
            pts = [
                (int(shoulder[0]*w), int(shoulder[1]*h)),
                (int(elbow[0]*w), int(elbow[1]*h)),
                (int(wrist[0]*w), int(wrist[1]*h))
            ]

            cv2.line(frame, pts[0], pts[1], (255, 0, 255), 4)
            cv2.line(frame, pts[1], pts[2], (255, 0, 255), 4)

            for p in pts:
                cv2.circle(frame, p, 8, (0,255,255), -1)

            # -------- Progress --------
            draw_progress_bar(frame, angle)

    except:
        pass

    # -------- UI --------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (640,80), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # REP COUNT
    cv2.putText(frame, f"{counter}",
                (270,65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 4)

    cv2.putText(frame, "REPS",
                (260,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # FEEDBACK BOX
    if "Correct" in feedback:
        color = (0,255,0)
    elif "Wrong" in feedback:
        color = (0,0,255)
    else:
        color = (200,200,200)

    cv2.rectangle(frame, (400,10), (630,70), color, -1)

    cv2.putText(frame, feedback,
                (410,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # LEFT INFO
    try:
        cv2.putText(frame, f"Arm: {arm}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"Angle: {int(angle)}",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    except:
        pass

    cv2.imshow("AI Physio Trainer", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# -------- Cleanup --------
cap.release()
cv2.destroyAllWindows()