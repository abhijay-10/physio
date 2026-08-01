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
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed: self.frame = frame

    def read(self):
        with self.read_lock:
            if self.grabbed: return self.frame.copy()
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
    """Calculates the angle at point b (Elbow joint flexion angle)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# 3. CONFIGURATION & MODELS
# ==========================================
CAMERA_INDEX = 2  
BOX_SIZE = 250  
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

print("🚀 Jones View Clinical Rotation Engine Active...")

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

        status = "PLACE AP ACUTE ARM ON TABLE"
        color = (0, 0, 255)
        side = "SCANNING..."
        checks = []

        if result.pose_landmarks:
            smoothed_landmarks = pose_stabilizer.smooth(result.pose_landmarks[0])

            # Determine closest visible arm
            left_score = smoothed_landmarks[11].visibility + smoothed_landmarks[13].visibility + smoothed_landmarks[15].visibility
            right_score = smoothed_landmarks[12].visibility + smoothed_landmarks[14].visibility + smoothed_landmarks[16].visibility

            if left_score > right_score:
                s_lm, e_lm, w_lm = smoothed_landmarks[11], smoothed_landmarks[13], smoothed_landmarks[15]
                pinky_lm, thumb_lm = smoothed_landmarks[17], smoothed_landmarks[21]
                side = "JONES VIEW: LEFT ELBOW"
            else:
                s_lm, e_lm, w_lm = smoothed_landmarks[12], smoothed_landmarks[14], smoothed_landmarks[16]
                pinky_lm, thumb_lm = smoothed_landmarks[18], smoothed_landmarks[22]
                side = "JONES VIEW: RIGHT ELBOW"

            # Translate landmarks to screen pixels
            shoulder_pt = (int((1 - s_lm.x) * w), int(s_lm.y * h))
            elbow_pt = (int((1 - e_lm.x) * w), int(e_lm.y * h))
            wrist_pt = (int((1 - w_lm.x) * w), int(w_lm.y * h))

            # Center target box directly over the Olecranon Process
            target_x = elbow_pt[0] - (BOX_SIZE // 2)
            target_y = elbow_pt[1] - (BOX_SIZE // 2)

            # --- CLINICAL THEORY VERIFICATION CHECKS ---
            
            # Check 1: Acute Flexion Angle Validation (Target: 25° to 65°)
            acute_angle = calculate_angle_3pt(shoulder_pt, elbow_pt, wrist_pt)
            angle_ok = 25.0 <= acute_angle <= 65.0
            checks.append((f"Acute Flexion ({acute_angle:.1f} Deg)", angle_ok))

            # Check 2: Anatomical AP Palm Up Orientation (Prevents Pronation/Arm Rotation)
            # When the palm faces up towards the camera in an AP view, the thumb landmark's 
            # horizontal relation shifts outwards relative to the pinky finger base.
            is_rotated = abs(thumb_lm.x - pinky_lm.x) < 0.015
            palm_up = (thumb_lm.y < w_lm.y) and not is_rotated
            checks.append(("Anatomical AP (Palm Up)", palm_up))

            # Determine posture status based purely on theory rules
            good_posture = angle_ok and palm_up
            if good_posture:
                status = "RIGHT POSTURE"
                color = (0, 255, 0)
            else:
                status = "WRONG POSTURE"
                color = (0, 0, 255)

            # Draw target box following the elbow joint
            cv2.rectangle(display_frame, (target_x, target_y), (target_x + BOX_SIZE, target_y + BOX_SIZE), color, 2, cv2.LINE_AA)

            # Central Ray Axis Crosshair at the center of the joint space
            cv2.line(display_frame, (elbow_pt[0] - 45, elbow_pt[1]), (elbow_pt[0] + 45, elbow_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (elbow_pt[0], elbow_pt[1] - 45), (elbow_pt[0], elbow_pt[1] + 45), (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Stabilized Skeleton Paths
            cv2.line(display_frame, shoulder_pt, elbow_pt, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, elbow_pt, wrist_pt, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(display_frame, shoulder_pt, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(display_frame, elbow_pt, 8, color, -1, cv2.LINE_AA)   
            cv2.circle(display_frame, wrist_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

            # On-Screen HUD Text
            cv2.putText(display_frame, side, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

            y_offset = 130
            for label, passed in checks:
                symbol = "OK" if passed else "X"
                chk_color = (0, 255, 0) if passed else (0, 0, 255)
                cv2.putText(display_frame, f"[{symbol}] {label}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, chk_color, 2, cv2.LINE_AA)
                y_offset += 40

            # Direct Contextual Guidance Text Block
            if not angle_ok:
                cv2.putText(display_frame, "👉 Flex the elbow fully to superimpose the bones.", (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            elif not palm_up:
                cv2.putText(display_frame, "👉 Fix Rotation: Face palm UP completely.", (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "✅ CENTRAL RAY ALIGNED: READY TO SCAN DISTAL HUMERUS", (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Acute Flexion Elbow Assistant", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    vs.stop()
    cv2.destroyAllWindows()
    detector.close()