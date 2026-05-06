# import streamlit as st
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import joblib
# import time

# # -------- LOAD MODEL --------
# model = joblib.load("pa_model.pkl")

# # Debug: check expected features
# st.write("Model expects features:", model.n_features_in_)

# # -------- CONFIG --------
# MODEL_PATH = "hand_landmarker.task"

# # -------- HAND CONNECTIONS --------
# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (5,9),(9,10),(10,11),(11,12),
#     (9,13),(13,14),(14,15),(15,16),
#     (13,17),(17,18),(18,19),(19,20),
#     (0,17)
# ]

# # -------- MEDIAPIPE SETUP --------
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=1
# )

# detector = vision.HandLandmarker.create_from_options(options)

# # -------- STREAMLIT UI --------
# st.title("🖐️ PA Hand Positioning Detector")
# st.write("Detect Correct / Wrong Hand Position in Real-Time")

# # Camera selector
# camera_index = st.selectbox("Select Camera", [0, 1, 2, 3], index=1)

# run = st.checkbox("Start Camera")

# frame_placeholder = st.empty()

# timestamp = 0

# # -------- MAIN LOOP --------
# if run:
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

#     if not cap.isOpened():
#         st.error(f"❌ Camera {camera_index} not working")
#         st.stop()

#     time.sleep(1)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera not working")
#             break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#         result = detector.detect_for_video(mp_image, timestamp)
#         timestamp += 1

#         if result.hand_landmarks:
#             for landmarks in result.hand_landmarks:

#                 # -------- DRAW CONNECTIONS --------
#                 for connection in HAND_CONNECTIONS:
#                     x1 = int(landmarks[connection[0]].x * w)
#                     y1 = int(landmarks[connection[0]].y * h)
#                     x2 = int(landmarks[connection[1]].x * w)
#                     y2 = int(landmarks[connection[1]].y * h)

#                     cv2.line(frame, (x1, y1), (x2, y2), (0,255,255), 2)

#                 # -------- DRAW POINTS --------
#                 for lm in landmarks:
#                     x = int(lm.x * w)
#                     y = int(lm.y * h)
#                     cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

#                 # -------- FEATURES (FIXED: NOW 7 FEATURES) --------
#                 data_row = [
#                     landmarks[0].z,
#                     landmarks[8].z,
#                     landmarks[12].z,
#                     landmarks[16].z,
#                     landmarks[20].z,
#                     abs(landmarks[2].x - landmarks[17].x),
#                     landmarks[0].x   # ✅ added feature to match model
#                 ]

#                 # Debug check
#                 # st.write(len(data_row))

#                 # -------- PREDICTION --------
#                 pred = model.predict([data_row])

#                 if pred[0] == 1:
#                     label = "CORRECT"
#                     color = (0,255,0)
#                 else:
#                     label = "WRONG"
#                     color = (0,0,255)

#                 cv2.putText(frame, label, (50,100),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

#         frame_placeholder.image(frame, channels="BGR")

#     cap.release()



import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import time

# -------- LOAD MODEL --------
model = joblib.load("pa_model.pkl")
st.write("Model expects features:", model.n_features_in_)

# -------- CONFIG --------
MODEL_PATH = "hand_landmarker.task"

# -------- HAND CONNECTIONS --------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# -------- STORE LAST CORRECT HAND --------
last_correct_hand = None

# -------- STYLED DRAW FUNCTION --------
def draw_styled_hand(frame, landmarks, h, w, color=(0,255,255)):
    overlay = frame.copy()

    # Draw connections
    for c in HAND_CONNECTIONS:
        x1 = int(landmarks[c[0]].x * w)
        y1 = int(landmarks[c[0]].y * h)
        x2 = int(landmarks[c[1]].x * w)
        y2 = int(landmarks[c[1]].y * h)

        cv2.line(overlay, (x1,y1), (x2,y2), color, 4)

    # Draw glowing joints
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(overlay, (x,y), 10, color, -1)
        cv2.circle(frame, (x,y), 4, (255,255,255), -1)

    # Blend
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

# -------- MEDIAPIPE SETUP --------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -------- STREAMLIT UI --------
st.title("🖐️ PA Hand Positioning Detector (Pro)")
st.write("Real-time posture detection with correction")

camera_index = st.selectbox("Select Camera", [0,1,2,3], index=1)
run = st.checkbox("Start Camera")

frame_placeholder = st.empty()
timestamp = 0

# -------- MAIN LOOP --------
if run:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(f"❌ Camera {camera_index} not working")
        st.stop()

    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:

                # -------- DRAW USER HAND --------
                draw_styled_hand(frame, landmarks, h, w, (0,200,255))

                # -------- FEATURES --------
                data_row = [
                    landmarks[0].z,
                    landmarks[8].z,
                    landmarks[12].z,
                    landmarks[16].z,
                    landmarks[20].z,
                    abs(landmarks[2].x - landmarks[17].x),
                    landmarks[0].x
                ]

                # -------- PREDICTION --------
                pred = model.predict([data_row])

                if pred[0] == 1:
                    label = "CORRECT"
                    color = (0,255,0)

                    # ✅ SAFE COPY (important)
                    last_correct_hand = [lm for lm in landmarks]

                else:
                    label = "WRONG"
                    color = (0,0,255)

                    # 🔥 DRAW LAST CORRECT HAND
                    if last_correct_hand is not None:
                        draw_styled_hand(frame, last_correct_hand, h, w, (0,255,0))

                        cv2.putText(frame, "MATCH THIS POSITION",
                                    (40,150),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0,255,0),
                                    2)

                # -------- LABEL --------
                cv2.putText(frame, label,
                            (40,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            color,
                            3)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()