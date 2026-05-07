import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# PAGE CONFIG
# ==========================================
# st.set_page_config(
#     page_title="Hand Posture Detection",
#     layout="wide"
# )

st.title("🖐️ Hand Posture Detection")

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = joblib.load("obliquehand/hand_model.pkl")
label_encoder = joblib.load("obliquehand/label_encoder.pkl")

# ==========================================
# LOAD MEDIAPIPE MODEL
# ==========================================
MODEL_PATH = "D:\\physio\\obliquehand\\hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
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
camera_index = 1

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():

    st.error("❌ Camera not opening")
    st.stop()

frame_placeholder = st.empty()

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

    prediction_text = "No Hand"

    # ======================================
    # DETECT HAND
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

            # --------------------------------
            # PREDICTION
            # --------------------------------
            X = pd.DataFrame([row])

            prediction = model.predict(X)[0]

            label = label_encoder.inverse_transform(
                [prediction]
            )[0]

            prediction_text = label

            # --------------------------------
            # COLORS
            # --------------------------------
            if label == "Left":

                color = (0,255,0)

                message = "✅ Correct LEFT Posture"

            elif label == "Right":

                color = (255,0,0)

                message = "✅ Correct RIGHT Posture"

            else:

                color = (0,0,255)

                message = "❌ Wrong Posture"

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
                    color,
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
                    color,
                    -1
                )

            # --------------------------------
            # SHOW TEXT
            # --------------------------------
            cv2.putText(
                frame,
                message,
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

    # ======================================
    # DISPLAY FRAME
    # ======================================
    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )

# ==========================================
# RELEASE
# ==========================================
cap.release()

# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import joblib
# import time

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # ==========================================
# # PAGE CONFIG
# # ==========================================
# st.set_page_config(
#     page_title="Hand Posture Detection",
#     layout="wide"
# )

# st.title("🖐️ Hand Posture Detection")

# # ==========================================
# # SIDEBAR - CAMERA SETTINGS
# # ==========================================
# st.sidebar.header("Settings")
# # Allows you to switch to your external camera
# camera_index = st.sidebar.selectbox("Select Camera Index", options=[0, 1, 2], index=0)
# run_detection = st.sidebar.checkbox("Start Detection", value=True)

# # ==========================================
# # LOAD MODELS (Cached for performance)
# # ==========================================
# @st.cache_resource
# def load_assets():
#     model = joblib.load("hand_model.pkl")
#     label_encoder = joblib.load("label_encoder.pkl")
    
#     MODEL_PATH = "hand_landmarker.task"
#     base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=1,
#         min_hand_detection_confidence=0.5,
#         min_hand_presence_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     detector = vision.HandLandmarker.create_from_options(options)
#     return model, label_encoder, detector

# model, label_encoder, detector = load_assets()

# # Hand Connections
# HAND_CONNECTIONS = [
#     (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
#     (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
#     (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
# ]

# # ==========================================
# # CAMERA INITIALIZATION
# # ==========================================
# cap = cv2.VideoCapture(camera_index)

# if not cap.isOpened():
#     st.error(f"❌ Camera at index {camera_index} not opening. Try another index.")
#     st.stop()

# frame_placeholder = st.empty()

# # ==========================================
# # MAIN LOOP
# # ==========================================
# try:
#     while run_detection:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Unable to read from camera. Check connection.")
#             break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb
#         )

#         result = detector.detect(mp_image)
        
#         # Default if no hand is detected
#         message = "No Hand Detected"
#         color = (255, 255, 255)

#         if result.hand_landmarks:
#             for hand_landmarks in result.hand_landmarks:
#                 row = []
#                 points = []

#                 for lm in hand_landmarks:
#                     row.extend([lm.x, lm.y, lm.z])
#                     points.append((int(lm.x * w), int(lm.y * h)))

#                 # PREDICTION
#                 X = pd.DataFrame([row])
#                 prediction = model.predict(X)[0]
#                 label = label_encoder.inverse_transform([prediction])[0]

#                 # LOGIC: Correct vs Wrong
#                 # Assuming your model classes are "Left" and "Right"
#                 if label in ["Left", "Right"]:
#                     color = (0, 255, 0) # Green
#                     message = f"✅ Correct: {label}"
#                 else:
#                     color = (0, 0, 255) # Red
#                     message = "❌ Wrong Posture"

#                 # DRAW SKELETON
#                 for connection in HAND_CONNECTIONS:
#                     x1, y1 = points[connection[0]]
#                     x2, y2 = points[connection[1]]
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 3)

#                 for pt in points:
#                     cv2.circle(frame, pt, 5, (255, 255, 255), -1)

#         # OVERLAY TEXT
#         cv2.putText(frame, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#         # DISPLAY FRAME
#         # Converting BGR to RGB for Streamlit display
#         frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#         # CRITICAL: This sleep prevents the flickering/stuttering
#         time.sleep(0.01)

# finally:
#     cap.release()
#     cv2.destroyAllWindows()