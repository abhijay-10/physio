import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a, b, c are (x, y) coordinates.
    b is the vertex (the joint we are measuring).
    """
    a = np.array(a)  # First point (e.g., Shoulder)
    b = np.array(b)  # Mid point (e.g., Hip)
    c = np.array(c)  # End point (e.g., Knee)

    # Calculate the vectors
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is the smaller interior angle
    if angle > 180.0:
        angle = 360 - angle

    return angle

def extract_landmarks(landmarks):
    """
    Converts MediaPipe landmarks into a dictionary for easy access.
    Accepts the landmarks list from the pose detector.
    """
    # Create a dictionary to map index numbers to names
    lm_dict = {
        "nose": [landmarks[0].x, landmarks[0].y],
        "left_eye": [landmarks[2].x, landmarks[2].y],
        "right_eye": [landmarks[5].x, landmarks[5].y],
        "left_shoulder": [landmarks[11].x, landmarks[11].y],
        "right_shoulder": [landmarks[12].x, landmarks[12].y],
        "left_elbow": [landmarks[13].x, landmarks[13].y],
        "right_elbow": [landmarks[14].x, landmarks[14].y],
        "left_wrist": [landmarks[15].x, landmarks[15].y],
        "right_wrist": [landmarks[16].x, landmarks[16].y],
        "left_hip": [landmarks[23].x, landmarks[23].y],
        "right_hip": [landmarks[24].x, landmarks[24].y],
        "left_knee": [landmarks[25].x, landmarks[25].y],
        "right_knee": [landmarks[26].x, landmarks[26].y],
        "left_ankle": [landmarks[27].x, landmarks[27].y],
        "right_ankle": [landmarks[28].x, landmarks[28].y],
    }
    return lm_dict