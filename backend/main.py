from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from fastapi.staticfiles import StaticFiles
os.environ["GLOG_minloglevel"] = "3"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys
import asyncio
import threading
import queue
import cv2
import time

app = FastAPI(
    title="Physio Master Backend API",
    description="FastAPI backend for the Axoris Physio Master AI.",
    version="1.0.1" # Bumped version to force reload
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount captures directory
if not os.path.exists("captures"):
    os.makedirs("captures")
app.mount("/captures", StaticFiles(directory="captures"), name="captures")

# Global streaming state
active_stream_queue = None
active_thread = None
active_stop_event = None
global_requested_camera = 2 # Default to DroidCam as requested
camera_lock = threading.Lock()
active_cameras = []
global_telemetry = {"status": "calibrating", "message": "Analyzing posture...", "accuracy": 0}

# =========================================================
# MONKEY-PATCHING ENGINE
# This intercepts OpenCV and Streamlit outputs transparently
# =========================================================

# 1. Intercept Streamlit
class FakeEmpty:
    def image(self, img, channels="RGB"):
        global active_stream_queue, global_telemetry
        if channels == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # Autonomous Clinical Capture
        if global_telemetry.get("capture_requested"):
            filename = f"capture_{int(time.time())}.jpg"
            filepath = os.path.join("captures", filename)
            
            # Crop exactly the middle 800x600 of the 1280x720 frame to focus on the joint
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            cw, ch = 400, 300
            cropped = img[max(0, cy-ch):min(h, cy+ch), max(0, cx-cw):min(w, cx+cw)]
            
            cv2.imwrite(filepath, cropped)
            global_telemetry['capture_requested'] = False
            global_telemetry['last_capture_url'] = f"http://127.0.0.1:8000/captures/{filename}"
            print(f"[Capture] Saved clinical report snapshot to {filepath}")

        ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ret and active_stream_queue is not None:
            if active_stream_queue.full():
                try: active_stream_queue.get_nowait()
                except: pass
            active_stream_queue.put(buffer.tobytes())
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class FakeSidebar:
    def header(self, *args, **kwargs): pass
    def selectbox(self, label, options, index=0, **kwargs): return options[index]
    def checkbox(self, label, value=True, **kwargs): return True
    def button(self, *args, **kwargs): return False
    def radio(self, label, options, index=0, **kwargs): return options[index]
    def toggle(self, label, value=True, **kwargs): return value
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class SessionState(dict):
    def __getattr__(self, key):
        return self.get(key, None)
    def __setattr__(self, key, value):
        self[key] = value

class FakeST:
    def __init__(self):
        self.sidebar = FakeSidebar()
        self.session_state = SessionState()
    def set_page_config(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def header(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def cache_resource(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(func):
            return func
        return wrapper
    def empty(self): return FakeEmpty()
    def error(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def success(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def subheader(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def selectbox(self, label, options, index=0, **kwargs): return options[index]
    def radio(self, label, options, index=0, **kwargs): return options[index]
    def toggle(self, label, value=True, **kwargs): return value
    def button(self, *args, **kwargs): return False
    def checkbox(self, label, value=True, **kwargs): return True
    def progress(self, *args, **kwargs): return FakeEmpty()
    def container(self, *args, **kwargs): return type("Container", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None, "write": lambda *a, **k: None, "__getattr__": lambda s, n: lambda *a, **k: None})()
    def columns(self, spec, *args, **kwargs): 
        # return a list of FakeEmpty objects simulating columns
        num = spec if isinstance(spec, int) else len(spec)
        return [FakeEmpty() for _ in range(num)]
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return lambda *args, **kwargs: None

sys.modules['streamlit'] = FakeST()

# 2. Intercept OpenCV cv2.imshow
original_imshow = cv2.imshow
def fake_imshow(winname, mat):
    global active_stream_queue
    ret, buffer = cv2.imencode('.jpg', mat, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ret and active_stream_queue is not None:
        if active_stream_queue.full():
            try: active_stream_queue.get_nowait()
            except: pass
        active_stream_queue.put(buffer.tobytes())
cv2.imshow = fake_imshow

# 3. Intercept OpenCV input loops to allow safe termination
def fake_waitKey(delay=0):
    if active_stop_event is not None and active_stop_event.is_set():
        return 27 # ESC key to break standard OpenCV loops
    return -1
cv2.waitKey = fake_waitKey

original_VideoCapture = cv2.VideoCapture
class SuperFakeVideoCapture:
    def __init__(self, *args, **kwargs):
        global active_stop_event, global_requested_camera, camera_lock, active_cameras
        self.my_stop_event = active_stop_event
        self.cap = None
        self.latest_frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)
        self.holds_camera_lock = False
        
        active_cameras.append(self)
        print(f"[Lifecycle] SuperFakeVideoCapture created. Active cameras: {len(active_cameras)}")
        
        # Build test sequence of camera indices to try:
        # 1. Start with global_requested_camera (selected in the frontend)
        test_sequence = [global_requested_camera]
        
        # 2. Add the index requested by the script (args[0]) if different and valid
        if len(args) > 0 and isinstance(args[0], int) and args[0] not in test_sequence:
            test_sequence.append(args[0])
            
        # 3. Add other common camera indices as fallbacks
        for fallback_idx in [0, 1, 2, 3]:
            if fallback_idx not in test_sequence:
                test_sequence.append(fallback_idx)
        
        # Acquire camera lock safely before trying to open the camera.
        while not camera_lock.acquire(timeout=0.1):
            if self.my_stop_event is not None and self.my_stop_event.is_set():
                print(f"[Lifecycle] SuperFakeVideoCapture aborted creation: stop event set before lock acquisition.")
                return
        
        self.holds_camera_lock = True
        print(f"[Lifecycle] SuperFakeVideoCapture successfully acquired camera_lock. Probing sequence: {test_sequence}")
        
        for idx in test_sequence:
            if self.my_stop_event is not None and self.my_stop_event.is_set():
                break
                
            # We try opening the camera with DirectShow first (recommended for Windows), then standard backend.
            # For each backend, we try 1280x720 resolution first, then fall back to native resolution.
            opened_cap = None
            
            # Helper to try a specific configuration
            def try_config(use_dshow, set_resolution):
                c = None
                try:
                    if use_dshow:
                        c = original_VideoCapture(idx, cv2.CAP_DSHOW)
                    else:
                        c = original_VideoCapture(idx)
                except Exception as e:
                    print(f"Failed to instantiate VideoCapture({idx}, dshow={use_dshow}): {e}")
                    return None
                    
                if c is None or not c.isOpened():
                    if c is not None:
                        c.release()
                    return None
                    
                # Optimize camera configuration parameters
                try:
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass
                try:
                    c.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if set_resolution:
                    try:
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    except Exception:
                        pass
                
                # Test if the camera actually retrieves frames
                ret = False
                frame = None
                for _ in range(10): # up to 0.5s total wait
                    if self.my_stop_event is not None and self.my_stop_event.is_set():
                        break
                    ret, frame = c.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.05)
                    
                if ret and frame is not None:
                    return c
                else:
                    c.release()
                    return None

            # 1. Try DirectShow + HD (1280x720)
            opened_cap = try_config(use_dshow=True, set_resolution=True)
            
            # 2. Try DirectShow + Native Resolution
            if opened_cap is None and not (self.my_stop_event is not None and self.my_stop_event.is_set()):
                opened_cap = try_config(use_dshow=True, set_resolution=False)
                
            # 3. Try Standard (MSMF) + HD (1280x720)
            if opened_cap is None and not (self.my_stop_event is not None and self.my_stop_event.is_set()):
                opened_cap = try_config(use_dshow=False, set_resolution=True)
                
            # 4. Try Standard (MSMF) + Native Resolution
            if opened_cap is None and not (self.my_stop_event is not None and self.my_stop_event.is_set()):
                opened_cap = try_config(use_dshow=False, set_resolution=False)
                
            if opened_cap is not None:
                self.cap = opened_cap
                # Get the first valid frame
                ret, frame = self.cap.read()
                self.ret = ret
                self.latest_frame = frame
                print(f"Successfully opened camera {idx} using optimal configuration.")
                
                # Start background thread to flush buffer
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                break
                
        # If camera failed to open, release the lock.
        if not self.cap:
            print("[Lifecycle] Failed to open any camera. Cleaning up lock if held.")
            try:
                if self.holds_camera_lock:
                    camera_lock.release()
                    self.holds_camera_lock = False
            except RuntimeError:
                pass

    def _update(self):
        target_interval = 1.0 / 30.0  # Limit background capture thread to 30 FPS to save CPU/GIL
        while self.running and self.cap and self.cap.isOpened():
            if self.my_stop_event is not None and self.my_stop_event.is_set():
                break
            start_time = time.time()
            ret = self.cap.grab()
            if ret:
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = True
                        self.latest_frame = frame
                    # Push latest frame to queue (replace old frame if any)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put((True, frame))
            else:
                with self.lock:
                    self.ret = False
                time.sleep(0.01)  # prevent tight loop on capture failures
                continue
                
            elapsed = time.time() - start_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        if not self.cap: return False, None
        if self.my_stop_event is not None and self.my_stop_event.is_set():
            return False, None # Force loop break
            
        try:
            # Wait up to 30ms for a fresh frame
            ret, frame = self.frame_queue.get(timeout=0.03)
            if ret and frame is not None:
                return True, frame.copy()
            return False, None
        except queue.Empty:
            # Fall back to returning the last successfully read frame to prevent UI freezes/stuttering
            with self.lock:
                if self.latest_frame is not None:
                    return self.ret, self.latest_frame.copy()
                return False, None

    def set(self, propId, value):
        if not self.cap: return False
        return self.cap.set(propId, value)
        
    def release(self):
        global camera_lock, active_cameras
        self.running = False
        print("[Lifecycle] SuperFakeVideoCapture release() called. Waiting for background thread to exit...")
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print("[Lifecycle] WARNING: Background thread failed to join within 1.0s. It may be deadlocked.")
            else:
                print("[Lifecycle] Background thread exited cleanly.")
        
        if self.cap:
            print("[Lifecycle] Releasing hardware VideoCapture.")
            self.cap.release()
            self.cap = None
            
        if getattr(self, 'holds_camera_lock', False):
            print("[Lifecycle] Releasing camera_lock.")
            try:
                camera_lock.release()
            except RuntimeError as e:
                print(f"[Lifecycle] Warning: RuntimeError on camera_lock.release(): {e}")
            self.holds_camera_lock = False
            
        if self in active_cameras:
            try: 
                active_cameras.remove(self)
                print(f"[Lifecycle] Removed from active_cameras. Remaining: {len(active_cameras)}")
            except ValueError: pass
        
    def isOpened(self):
        if self.my_stop_event is not None and self.my_stop_event.is_set():
            return False
        return self.cap is not None

cv2.VideoCapture = SuperFakeVideoCapture

# =========================================================

def background_runner(script_path, stop_event):
    print(f"Starting isolated script thread for: {script_path}")
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    import re
    # Auto-patch user scripts that use LiveVideoStream infinite loops without stop checks
    # Dynamically capture and preserve exact indentation
    code = re.sub(
        r"([ \t]+)if frame is None: continue", 
        r"\1if frame is None:\n\1    if 'active_stop_event' in globals() and active_stop_event.is_set(): break\n\1    continue",
        code
    )
        
    module_globals = {
        '__name__': '__main__',
        '__file__': script_path,
        'cv2': cv2, # Provide patched cv2 explicitly
        'active_stop_event': stop_event, # Inject stop event for the patch
        'global_telemetry': global_telemetry # Inject telemetry bridge
    }
    
    try:
        exec(code, module_globals)
        print("Script finished successfully.")
    except Exception as e:
        import traceback
        print(f"Isolated script thread terminated safely: {e}")
        print(traceback.format_exc())
        print(f"Isolated script thread terminated safely: {e}")
    finally:
        print(f"[Lifecycle] background_runner finally block executing. Cleaning up {len(active_cameras)} active cameras.")
        for cap in list(active_cameras):
            try: cap.release()
            except Exception as e: print(f"[Lifecycle] Exception during cleanup release: {e}")

def cleanup_session_state():
    st_mock = sys.modules.get('streamlit')
    if st_mock is not None and hasattr(st_mock, 'session_state'):
        for key, value in list(st_mock.session_state.items()):
            if hasattr(value, 'stop'):
                try: value.stop()
                except Exception as e: print(f"Error stopping stream: {e}")
            elif hasattr(value, 'release'):
                try: value.release()
                except Exception as e: print(f"Error releasing camera: {e}")
        st_mock.session_state.clear()

@app.get("/")
def root():
    return {"message": "Welcome to the Physio Master Backend API. The video_feed endpoint is active."}

@app.get("/stop_stream")
def stop_stream():
    global active_stop_event
    print("[API] /stop_stream called. Force halting active camera feed.")
    if active_stop_event is not None:
        active_stop_event.set()
    cleanup_session_state()
    return {"status": "stopped"}

@app.get("/telemetry")
async def get_telemetry():
    import time
    with open("telemetry_hit.log", "a") as f:
        f.write(f"Hit at {time.time()} with data: {global_telemetry}\n")
    return global_telemetry

@app.post("/capture")
async def request_capture():
    global global_telemetry
    global_telemetry['capture_requested'] = True
    return {"status": "Capture initiated"}

@app.get("/captures_list")
async def get_captures_list():
    try:
        if not os.path.exists("captures"):
            return {"images": []}
        files = os.listdir("captures")
        # Filter for jpgs and sort by latest first
        images = [f"http://127.0.0.1:8000/captures/{f}" for f in files if f.endswith(".jpg") or f.endswith(".png")]
        images.sort(reverse=True)
        return {"images": images}
    except Exception as e:
        return {"images": []}

from pydantic import BaseModel
import urllib.request
import urllib.error
import json

class ChatRequest(BaseModel):
    message: str
    current_page: str
    history: list

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_path = os.path.join(base_dir, "assistant_knowledge.md")
    
    system_prompt = "You are the PhysioMaster Product Assistant.\n"
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            system_prompt += f.read()
            
    system_prompt += f"\n\nCURRENT CONTEXT: The user is currently viewing the page: {req.current_page}. Tailor your response if relevant.\n"
    
    # Construct conversation context
    prompt = ""
    for msg in req.history:
        role = "User" if msg.get("sender") == "user" else "Assistant"
        prompt += f"{role}: {msg.get('text')}\n"
    prompt += f"User: {req.message}\nAssistant:"
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    
    request_obj = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        # Run in thread to not block async loop
        def fetch():
            with urllib.request.urlopen(request_obj, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        
        result = await asyncio.to_thread(fetch)
        return {"response": result.get("response", "I'm sorry, I couldn't generate a response.")}
    except Exception as e:
        print(f"Ollama Error: {e}")
        return {"response": "I'm sorry, the connection timed out or failed. Please ensure Ollama is running and the model is loaded."}

stream_setup_lock = asyncio.Lock()

@app.get("/video_feed")
async def video_feed(request: Request, module: str, cam: int = 2, target: str = None):
    """
    Expects module query param like: ?module=chest/back_pose&cam=2
    """
    global active_stream_queue, active_thread, active_stop_event, global_requested_camera
    
    async with stream_setup_lock:
        global_requested_camera = cam
        
        # Stop any currently active camera module
        if active_stop_event is not None:
            active_stop_event.set()
            
        # Clean up and stop any cached camera streams in streamlit session_state
        cleanup_session_state()
        if target:
            sys.modules.get('streamlit').session_state['target_leg'] = target
            
        if active_thread is not None and active_thread.is_alive():
            await asyncio.to_thread(active_thread.join, timeout=2.0) # wait up to 2 seconds for previous thread to exit and release camera
            
        # Start new queue and lifecycle event
        active_stream_queue = queue.Queue(maxsize=2)
        active_stop_event = threading.Event()
        
        global global_telemetry
        global_telemetry = {"status": "calibrating", "message": "Analyzing posture...", "accuracy": 0}
        
        # Path resolution mapping
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parts = module.split("/")
        category = parts[0]
        
        if category == "spine":
            folder_path = "spine"
            script_name = "spine_app.py"
        elif category == "lowerback":
            folder_path = "lower_back"
            script_name = "front_ap.py"
        else:
            folder_path = module # e.g. "chest/back_pose"
            script_name = "stream.py"
            
        script_path = os.path.join(base_dir, folder_path, script_name)
        
        if os.path.exists(script_path):
            active_thread = threading.Thread(
                target=background_runner, 
                args=(script_path, active_stop_event), 
                daemon=True
            )
            active_thread.start()
        else:
            return {"error": f"Script not found at {script_path}"}
            
        my_queue = active_stream_queue
        my_stop = active_stop_event
        
    async def frame_generator():
        try:
            while not my_stop.is_set():
                if await request.is_disconnected():
                    print("Request disconnected (async check)")
                    break
                try:
                    # Run my_queue.get in thread so it doesn't block the async event loop
                    frame_bytes = await asyncio.to_thread(my_queue.get, timeout=0.2)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except queue.Empty:
                    continue
        finally:
            print("Generator exited, setting stop event and cleaning up session state")
            my_stop.set()
            cleanup_session_state()
                
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
