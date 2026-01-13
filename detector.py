import cv2
import time
import requests
from face_engine import FaceEngine

CAMERA_INDEX = 0
SCALE = 0.5
COOLDOWN = 5
SHOW_WINDOW = True
TH_ACCEPT = 0.4
WEBHOOK_URL = "http://localhost:3000/face-event"

engine = FaceEngine()
engine.start_watcher()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_sent = {}

print("[INFO] Realtime face daemon running")

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

        if "box" not in r:
            continue

        x1, y1, x2, y2 = r["box"]

        if dist < TH_ACCEPT:
            color = (0, 0, 255)  # RED
            label = f"{name} ({dist:.3f})"
            status = "REJECT"
            
        else:
            color = (0, 255, 0)  # GREEN
            label = f"UNKNOWN ({dist:.3f})"
            status = "ACCEPT"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        print(f"[GATE] {status} | {label}")


        if name == "UNKNOWN":
            continue

        if now - last_sent.get(name, 0) < COOLDOWN:
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
            print("[WEBHOOK ERROR]", e)

    if SHOW_WINDOW:
        cv2.imshow("Face Recognition", frame)

    # tekan Q buat keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
