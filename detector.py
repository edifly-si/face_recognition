import cv2
import time
import requests
from face_engine import FaceEngine

CAMERA_INDEX = 0
SCALE = 0.5
COOLDOWN_SEC = 5
WEBHOOK_URL = "http://localhost:3000/face-event"

engine = FaceEngine()
engine.start_watcher()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_sent = {}

print("[INFO] Realtime daemon started")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    results = engine.recognize(frame)
    now = time.time()

    for r in results:
        name = r["name"]
        dist = r["distance"]
        
        print(f"[INFO] Detected: {name} (dist: {dist:.4f})")

        if name == "UNKNOWN":
            continue

        last = last_sent.get(name, 0)
        if now - last < COOLDOWN_SEC:
            continue

        payload = {
            "name": name,
            "distance": round(dist, 4),
            "timestamp": int(now)
        }

        try:
            # requests.post(WEBHOOK_URL, json=payload, timeout=2)
            last_sent[name] = now
        except Exception as e:
            print("[ERROR] Webhook failed:", e)

    time.sleep(0.05)
