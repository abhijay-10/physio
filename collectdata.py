import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7 # Helps keep skeleton stable during movement
)
mp_drawing = mp.solutions.drawing_utils

def run_collector():
    cap = cv2.VideoCapture(0)
    data = []
    recording_state = "IDLE" # IDLE, RIGHT, WRONG
    
    print("KEYS: [R] - Record Right | [W] - Record Wrong | [S] - Save & Exit | [Q] - Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # Flip the frame for mirror view (easier for self-posing)
        frame = cv2.flip(frame, 1) 
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        status_color = (255, 255, 255) # Default white
        overlay_color = (128, 128, 128) # Gray for connections

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                
                # --- THIS IS THE VISUAL FEEDBACK KEY ---
                # Define drawing style based on recording status
                if recording_state == "RIGHT":
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                elif recording_state == "WRONG":
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2)
                else:
                    # Default MediaPipe rainbow style when not recording
                    landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style()
                    connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style()

                # DRAW THE SKELETON
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec,
                    connection_drawing_spec
                )
                
                # Save data if in a recording state
                if recording_state in ["RIGHT", "WRONG"]:
                    res = []
                    for lm in hand_lms.landmark:
                        res.extend([lm.x, lm.y, lm.z])
                    res.append(recording_state) # Append the label
                    data.append(res)

        # UI Overlay
        if recording_state == "RIGHT": status_color = (0, 255, 0) # Green
        elif recording_state == "WRONG": status_color = (0, 0, 255) # Red
        
        cv2.rectangle(frame, (0, 0), (350, 110), (0,0,0), -1) # Background for text
        cv2.putText(frame, f"REC: {recording_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"Collected: {len(data)} frames", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('X-Ray Pose Trainer - Data Collection', frame)

        # Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording_state = "RIGHT"
        elif key == ord('w'):
            recording_state = "WRONG"
        elif key == ord('s'):
            save_data(data)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_data(data):
    if not data:
        print("No data collected.")
        return
    
    # Create columns: x0, y0, z0, x1, y1, z1 ... x20, y20, z20, target
    columns = []
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])
    columns.append('target')
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('pa_hand_raw_dataset.csv', index=False)
    print(f"Successfully saved {len(data)} samples to 'pa_hand_raw_dataset.csv'")

if __name__ == "__main__":
    run_collector()