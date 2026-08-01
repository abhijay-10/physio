import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- THREADING VIDEO LOCK ---
class LiveVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

class PoseStabilizer:
    def __init__(self, alpha=0.35): 
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
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==========================================
# FUNCTIONAL ENTRY POINT FOR MASTER APP
# ==========================================
def run_spine_analysis():
    st.subheader("🦴 Live Spine Curvature Engine")
    
    camera_index = st.sidebar.selectbox("Select Side Camera Device", options=[0, 1, 2, 3], index=1, key="spine_cam_sel")
    sharpen_feed = st.sidebar.toggle("Auto-Sharpen (Anti-Blur)", value=True, key="spine_sharp_tgl")

    col1, col2 = st.columns([2.5, 1])
    with col2:
        cervical_box = st.empty()
        thoracic_box = st.empty()
        lean_box = st.empty()
        st.divider()
        p_bar = st.progress(0.0)
        p_txt = st.empty()

    with col1:
        frame_window = st.empty()

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker_full.task").replace("\\", "/")
    stabilizer = PoseStabilizer(alpha=0.35)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.35, min_tracking_confidence=0.45 
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    # Protects camera initialization states from memory stack leakage during page re-runs
    if "active_spine_camera" not in st.session_state:
        st.session_state.active_spine_camera = None
    if "current_spine_cam_idx" not in st.session_state:
        st.session_state.current_spine_cam_idx = -1

    if st.session_state.current_spine_cam_idx != camera_index:
        if st.session_state.active_spine_camera is not None:
            st.session_state.active_spine_camera.stop()
        st.session_state.active_spine_camera = LiveVideoStream(src=camera_index).start()
        st.session_state.current_spine_cam_idx = camera_index
        time.sleep(0.5)

    vs = st.session_state.active_spine_camera

    try:
        st.session_state.active_mod = "spine"
        while vs.started and st.session_state.active_mod is not None:
            frame = vs.read()
            if frame is None: continue
            
            if sharpen_feed:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                frame = cv2.filter2D(frame, -1, kernel)

            display_frame = cv2.flip(frame, 1)
            h, w, _ = display_frame.shape
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            current_timestamp_ms = int(time.time() * 1000)
            if \'last_timestamp_ms\' not in locals(): last_timestamp_ms = 0
            if current_timestamp_ms <= last_timestamp_ms: current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            
            current_landmarks = None
            if result.pose_landmarks:
                current_landmarks = stabilizer.smooth(result.pose_landmarks[0])

            if current_landmarks:
                pixel_pts = [(int((1 - p.x) * w), int(p.y * h)) for p in current_landmarks]
                ear, shoulder, hip = (7, 11, 23) if current_landmarks[11].visibility > current_landmarks[12].visibility else (8, 12, 24)

                c_ang = calculate_angle_3pt(pixel_pts[ear], pixel_pts[shoulder], pixel_pts[hip])
                # Use the same ear-shoulder-hip angle for thoracic evaluation
                # since the knee dependency breaks detection when sitting/lying down.
                t_ang = c_ang 

                # Calculate trunk tilt angle relative to vertical axis in pixel space
                trunk_dx = pixel_pts[hip][0] - pixel_pts[shoulder][0]
                trunk_dy = pixel_pts[hip][1] - pixel_pts[shoulder][1]
                trunk_tilt = np.degrees(np.arctan2(abs(trunk_dx), abs(trunk_dy) + 1e-6))
                
                # Check if trunk is straight erect (under 14 degrees tilt)
                trunk_erect = trunk_tilt <= 14.0

                # Check if head/neck is tilted backward (neck hyperextension / looking up)
                # 1. Nose is higher than the ear (looking up)
                nose_higher = current_landmarks[0].y < current_landmarks[ear].y
                
                # 2. Ear is behind the shoulder (based on facing direction)
                is_facing_right = pixel_pts[0][0] > pixel_pts[shoulder][0]
                if is_facing_right:
                    ear_behind_shoulder = pixel_pts[ear][0] < pixel_pts[shoulder][0] - 8
                else:
                    ear_behind_shoulder = pixel_pts[ear][0] > pixel_pts[shoulder][0] + 8
                    
                is_head_back = nose_higher or ear_behind_shoulder

                score = max(0.0, min(1.0, c_ang / 180.0))
                precision_percentage = score * 100
                is_valid_posture = precision_percentage >= 87.0 and trunk_erect and not is_head_back

                spine_color = (0, 255, 0) if is_valid_posture else (0, 0, 255)
                joint_dots_color = (255, 255, 255) if is_valid_posture else (0, 165, 255)

                cv2.line(display_frame, pixel_pts[ear], pixel_pts[shoulder], spine_color, 4, cv2.LINE_AA)
                cv2.line(display_frame, pixel_pts[shoulder], pixel_pts[hip], spine_color, 4, cv2.LINE_AA)

                for milestone in [ear, shoulder, hip]:
                    cv2.circle(display_frame, pixel_pts[milestone], 7, joint_dots_color, -1)

                # --- CERVICAL INSTRUCTIONS IF WRONG ---
                if is_head_back:
                    cervical_box.error(f"⚠️ NECK HYPEREXTENSION\n\n👉 **Instruction:** Keep your head level. Look straight ahead to correct backward head tilt.")
                elif c_ang >= 162.0: 
                    cervical_box.success(f"✅ CERVICAL: NORMAL ({c_ang:.1f}°)")
                else: 
                    cervical_box.error(f"⚠️ CERVICAL STRAIN ({c_ang:.1f}°)\n\n👉 **Instruction:** Pull your head back. Align your ears directly over your shoulders to fix forward head tilt.")

                # --- THORACIC INSTRUCTIONS IF WRONG ---
                if t_ang >= 168.0: 
                    thoracic_box.success(f"✅ THORACIC: NORMAL ({t_ang:.1f}°)")
                else: 
                    thoracic_box.error(f"⚠️ THORACIC KYPHOSIS ({t_ang:.1f}°)\n\n👉 **Instruction:** Roll your shoulders back and lift your chest. Avoid slouching your upper back forward.")

                # --- TRUNK LEAN INSTRUCTIONS IF WRONG ---
                if trunk_erect:
                    lean_box.success("✅ TRUNK: ERECT")
                else:
                    lean_box.error(f"⚠️ TRUNK LEAN ({trunk_tilt:.1f}°)\n\n👉 **Instruction:** Stand straight erect. Avoid leaning forward or backward.")

                if is_valid_posture:
                    cv2.rectangle(display_frame, (0, 0), (w, h), (0, 255, 0), 12)
                    cv2.putText(display_frame, "POSTURE ALIGNED", (w//2 - 180, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), 12)
                    warn_y = h - 40
                    if not trunk_erect:
                        cv2.putText(display_frame, "👉 Stand straight erect. Do not lean.", (30, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        warn_y -= 45
                    if t_ang < 168.0:
                        cv2.putText(display_frame, "👉 Roll shoulders back and lift chest.", (30, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        warn_y -= 45
                    if is_head_back:
                        cv2.putText(display_frame, "👉 Keep head level. Look straight ahead.", (30, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        warn_y -= 45
                    elif c_ang < 162.0:
                        cv2.putText(display_frame, "👉 Pull head back. Align ears over shoulders.", (30, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                # --- TELEMETRY updates for voice assistance ---
                if 'global_telemetry' in globals():
                    if is_valid_posture:
                        global_telemetry['status'] = "good"
                        global_telemetry['message'] = "Perfect alignment. Keep holding."
                        global_telemetry['accuracy'] = int(precision_percentage)
                    else:
                        global_telemetry['status'] = "bad"
                        global_telemetry['accuracy'] = int(precision_percentage)
                        if not trunk_erect:
                            global_telemetry['message'] = "Stand straight erect. Do not lean."
                        elif is_head_back:
                            global_telemetry['message'] = "Keep head level. Look straight ahead."
                        elif c_ang < 162.0:
                            global_telemetry['message'] = "Pull head back. Align ears over shoulders."
                        elif t_ang < 168.0:
                            global_telemetry['message'] = "Roll shoulders back and lift chest."
                        else:
                            global_telemetry['message'] = "Align your posture in camera view"

                st.session_state.spine_score = score
                p_bar.progress(score)
                p_txt.write(f"Spinal Alignment Precision: {precision_percentage:.1f}%")
            else:
                cervical_box.warning("🔎 SCANNING FOR PROFILE LENS INTERACTION...")
                thoracic_box.empty()
                lean_box.empty()
                p_bar.progress(0.0)
                if 'global_telemetry' in globals():
                    global_telemetry['status'] = "calibrating"
                    global_telemetry['message'] = "Align your posture in camera view"
                    global_telemetry['accuracy'] = 10

            frame_window.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    finally:
        detector.close()

if __name__ == "__main__":
    run_spine_analysis()