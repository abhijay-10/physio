import cv2
import numpy as np
import time
import csv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- Angle Function --------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# -------- Camera Setup --------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# -------- MediaPipe Setup --------
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

# -------- CSV Setup --------
file = open("exercise_data.csv", mode="w", newline="")
writer = csv.writer(file)
writer.writerow(["angle", "shoulder_movement", "rep_time", "label", "arm"])

# -------- Variables --------
stage = None
start_time = 0
last_rep_time = 0
cooldown = 1.0
timestamp = 0

print("\nC = Correct | W = Wrong | Q = Quit\n")

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

            # -------- BOTH ARMS --------
            # LEFT
            l_sh = lm[11]
            l_el = lm[13]
            l_wr = lm[15]

            # RIGHT
            r_sh = lm[12]
            r_el = lm[14]
            r_wr = lm[16]

            # Visibility
            left_vis = min(l_sh.visibility, l_el.visibility, l_wr.visibility)
            right_vis = min(r_sh.visibility, r_el.visibility, r_wr.visibility)

            # Select best arm
            if right_vis > left_vis:
                shoulder = [r_sh.x, r_sh.y]
                elbow = [r_el.x, r_el.y]
                wrist = [r_wr.x, r_wr.y]
                arm_used = "RIGHT"
            else:
                shoulder = [l_sh.x, l_sh.y]
                elbow = [l_el.x, l_el.y]
                wrist = [l_wr.x, l_wr.y]
                arm_used = "LEFT"

            # -------- Angle --------
            angle = calculate_angle(shoulder, elbow, wrist)

            pts = [
                (int(shoulder[0]*w), int(shoulder[1]*h)),
                (int(elbow[0]*w), int(elbow[1]*h)),
                (int(wrist[0]*w), int(wrist[1]*h))
            ]

            # -------- Skeleton --------
            cv2.line(frame, pts[0], pts[1], (0,255,255), 3)
            cv2.line(frame, pts[1], pts[2], (0,255,255), 3)
            for p in pts:
                cv2.circle(frame, p, 6, (0,0,255), -1)

            # -------- Display --------
            cv2.putText(frame, f"Angle: {int(angle)}", pts[1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(frame, f"Arm: {arm_used}",
                        (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

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

                    cv2.putText(frame, "Press C/W",
                                (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    cv2.imshow("Data Collection", frame)

                    key = cv2.waitKey(0)

                    if key == ord('c'):
                        writer.writerow([angle, shoulder_move, rep_time, 1, arm_used])
                        print("Saved: Correct")

                    elif key == ord('w'):
                        writer.writerow([angle, shoulder_move, rep_time, 0, arm_used])
                        print("Saved: Wrong")

    except:
        pass

    # -------- UI --------
    cv2.rectangle(frame, (0,0), (640,35), (0,0,0), -1)
    cv2.putText(frame, "Do curls | C=Correct W=Wrong Q=Quit",
                (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# -------- Cleanup --------
cap.release()
file.close()
cv2.destroyAllWindows()

print("✅ Data saved to exercise_data.csv")