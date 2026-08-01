import os
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. HARDWARE THREADING LOCK (Zero Lag Buffer)
# ==========================================
class LiveVideoStream:
    def __init__(self, src=2):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.started = False

    def start(self):
        self.started = True
        return self

    def read(self):
        if not self.started: return None
        grabbed, frame = self.stream.read()
        if grabbed and frame is not None:
            return frame
        return None

    def stop(self):
        self.started = False
        if self.stream.isOpened(): self.stream.release()

# ==========================================
# 2. RAW COORDINATE FILTER (Prevents Jitter)
# ==========================================
class PoseStabilizer:
    def __init__(self, alpha=0.30): 
        self.alpha = alpha
        self.prev_l = None

    def smooth(self, current_l):
        if self.prev_l is None:
            self.prev_l = current_l
            return current_l
        smoothed = []
        for p, c in zip(self.prev_l, current_l):
            s_pt = type(c)(
                x = p.x * (1 - self.alpha) + c.x * self.alpha,
                y = p.y * (1 - self.alpha) + c.y * self.alpha,
                z = p.z * (1 - self.alpha) + c.z * self.alpha,
                visibility = c.visibility
            )
            smoothed.append(s_pt)
        self.prev_l = smoothed
        return smoothed

def calculate_angle_3pt(a, b, c):
    """Calculates the extension angle at the elbow vertex (point b)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 3. CONFIGURATION & MODELS
# ==========================================
CAMERA_INDEX = 2  
BOX_SIZE = 250  # Matches the square collimation area over the joint space
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker_full.task").replace("\\", "/")

pose_stabilizer = PoseStabilizer(alpha=0.28)

# ==========================================
# 4. INITIALIZE MEDIAPIPE VISION TASK
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.25,
    min_tracking_confidence=0.45
)
detector = vision.PoseLandmarker.create_from_options(options)

vs = LiveVideoStream(src=CAMERA_INDEX).start()
time.sleep(1.0) 

print("🚀 Standard AP Fully Extended Elbow Scanner Online...")

# ==========================================
# 5. MAIN PROCESSING PIPELINE
# ==========================================
try:
    while vs.started:
        frame = vs.read()
        if frame is None: continue

        display_frame = cv2.flip(frame, 1)
        h, w, _ = display_frame.shape

        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        status = "PLACE EXTENDED AP ARM IN WORKSPACE"
        color = (0, 0, 255)
        side = "SCANNING..."
        checks = []

        if result.pose_landmarks:
            smoothed_landmarks = pose_stabilizer.smooth(result.pose_landmarks[0])

            # Select the main visible arm
            left_score = smoothed_landmarks[11].visibility + smoothed_landmarks[13].visibility + smoothed_landmarks[15].visibility
            right_score = smoothed_landmarks[12].visibility + smoothed_landmarks[14].visibility + smoothed_landmarks[16].visibility

            if left_score > right_score:
                s_lm, e_lm, w_lm = smoothed_landmarks[11], smoothed_landmarks[13], smoothed_landmarks[15]
                pinky_lm, thumb_lm = smoothed_landmarks[17], smoothed_landmarks[21]
                side = "STANDARD AP: LEFT ELBOW"
            else:
                s_lm, e_lm, w_lm = smoothed_landmarks[12], smoothed_landmarks[14], smoothed_landmarks[16]
                pinky_lm, thumb_lm = smoothed_landmarks[18], smoothed_landmarks[22]
                side = "STANDARD AP: RIGHT ELBOW"

            # Translate landmarks into pixel coordinates
            shoulder_pt = (int((1 - s_lm.x) * w), int(s_lm.y * h))
            elbow_pt = (int((1 - e_lm.x) * w), int(e_lm.y * h))
            wrist_pt = (int((1 - w_lm.x) * w), int(w_lm.y * h))

            # Center the dynamic tracking box directly over the elbow joint
            target_x = elbow_pt[0] - (BOX_SIZE // 2)
            target_y = elbow_pt[1] - (BOX_SIZE // 2)

            # Calculate real-time arm extension angle
            extension_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)

            # ==========================================
            # CLINICAL AP CRITERIA VALIDATION
            # ==========================================
            # Check 1: Full Extension Rule (Anatomical straight line window: 155° to 205°)
            extension_ok = 155.0 <= extension_angle <= 205.0
            checks.append((f"Elbow Fully Straight ({extension_angle:.1f} Deg)", extension_ok))

            # Check 2: Anatomical Hand Supination (Palm facing completely UP)
            # Verifies that the thumb landmark orientation is cleared horizontally relative to the wrist or pinky
            hand_supinated = (thumb_lm.y < w_lm.y + 0.15) or (thumb_lm.y < pinky_lm.y + 0.15)
            checks.append(("Hand Supinated (Palm UP)", hand_supinated))

            # Check 3: Horizontal Arm Alignment (Ensure the arm is extended horizontally)
            se_dx = elbow_pt[0] - shoulder_pt[0]
            se_dy = elbow_pt[1] - shoulder_pt[1]
            se_tilt = np.degrees(np.arctan2(abs(se_dy), abs(se_dx) + 1e-6))

            ew_dx = wrist_pt[0] - elbow_pt[0]
            ew_dy = wrist_pt[1] - elbow_pt[1]
            ew_tilt = np.degrees(np.arctan2(abs(ew_dy), abs(ew_dx) + 1e-6))

            arm_horizontal = se_tilt <= 8.0 and ew_tilt <= 8.0
            checks.append((f"Arm Horizontal (Tilted {max(se_tilt, ew_tilt):.1f} Deg)", arm_horizontal))

            # Verify if all parameters are satisfied
            good_posture = extension_ok and hand_supinated and arm_horizontal
            if good_posture:
                status = "RIGHT POSTURE"
                color = (0, 255, 0) # Green box lock
            else:
                status = "WRONG POSTURE"
                color = (0, 0, 255) # Red warning box

            # Draw the box following the straight joint space
            cv2.rectangle(
                display_frame,
                (target_x, target_y),
                (target_x + BOX_SIZE, target_y + BOX_SIZE),
                color,
                2,
                cv2.LINE_AA
            )

            # Central Ray Axis Crosshair at the center of the joint space matching the image layout
            cv2.line(display_frame, (elbow_pt[0] - 40, elbow_pt[1]), (elbow_pt[0] + 40, elbow_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (elbow_pt[0], elbow_pt[1] - 40), (elbow_pt[0], elbow_pt[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Stabilized Skeleton Vector Paths
            cv2.line(display_frame, shoulder_pt, elbow_pt, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.circle(display_frame, shoulder_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(display_frame, elbow_pt, 8, color, -1, cv2.LINE_AA)   
            cv2.circle(display_frame, wrist_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

            # HUD Display Elements Text Rendering
            cv2.putText(display_frame, side, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

            y_offset = 130
            for label, passed in checks:
                symbol = "OK" if passed else "X"
                chk_color = (0, 255, 0) if passed else (0, 0, 255)
                cv2.putText(display_frame, f"[{symbol}] {label}", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, chk_color, 2, cv2.LINE_AA)
                y_offset += 40

            if not extension_ok:
                cv2.putText(display_frame, "👉 Keep your arm horizontal and straight (180 degrees)", (20, y_offset + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            elif not arm_horizontal:
                cv2.putText(display_frame, "👉 Keep your arm horizontal and straight (180 degrees)", (20, y_offset + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            elif not hand_supinated:
                cv2.putText(display_frame, "👉 Rotate your wrist so your palm faces flat UP towards the ceiling", (20, y_offset + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "✅ CENTRAL RAY DETECTING JOINTS: READY TO CAPTURE", (20, y_offset + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("AP Extended Elbow Assistant", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    vs.stop()
    cv2.destroyAllWindows()
    detector.close()