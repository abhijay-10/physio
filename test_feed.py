import requests
import time
import threading

def run_feed():
    try:
        # Simulate browser requesting video_feed
        r = requests.get("http://127.0.0.1:8000/video_feed?module=hand/pa_hand", stream=True, timeout=5)
        for chunk in r.iter_content(chunk_size=1024):
            pass
    except Exception as e:
        print("Feed ended:", e)

threading.Thread(target=run_feed, daemon=True).start()

print("Waiting 3 seconds for video_feed to start and debug mutate to fire...")
time.sleep(3)

print("Fetching telemetry...")
try:
    r = requests.get("http://127.0.0.1:8000/telemetry", timeout=2)
    print("Telemetry Response:", r.json())
except Exception as e:
    print("Telemetry failed:", e)
