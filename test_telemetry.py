import requests
import time

print("Fetching telemetry...")
r = requests.get("http://127.0.0.1:8000/telemetry")
print("Response:", r.json())
