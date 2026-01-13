import cv2
import time
from face_engine import FaceEngine
import base64
from ws_client import WSClient
from settings import WS_URL, WS_JPEG_QUALITY

from settings import (
    VIDEO_SOURCE, SCALE, COOLDOWN,
    SHOW_WINDOW, TH_ACCEPT, WEBHOOK_URL
)

ws = WSClient(WS_URL) if WS_URL else None

engine = FaceEngine()
engine.start_watcher()

print("[INFO] Video source:", VIDEO_SOURCE)
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_sent = {}

print("[INFO] Realtime face daemon running")

def encode_frame(frame):
    ok, buf = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, WS_JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

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
        x1, y1, x2, y2 = r["box"]
        print("[DEBUG]", name, dist, (x1, y1, x2, y2))

        if dist > TH_ACCEPT:
            color = (0, 255, 0)
            label = f"PASS ({dist:.3f})"
            status = "PASS"
        else:
            color = (0, 0, 255)
            label = f"{name} ({dist:.3f})"
            status = "REJECT"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        print(f"[GATE] {status} | {label}")
        if ws:
            frame_b64 = encode_frame(frame)

            ws.send({
                "type": "face_event",
                "name": name,
                "distance": round(dist, 4),
                "status": status,
                "box": [x1, y1, x2, y2],
                "timestamp": int(now),
                "frame": frame_b64
            })

        if status != "ACCEPT":
            continue

        if now - last_sent.get(name, 0) < COOLDOWN:
            continue

        payload = {
            "name": name,
            "distance": round(dist, 4),
            "timestamp": int(now)
        }

        try:
            last_sent[name] = now
        except Exception as e:
            print("[WEBHOOK ERROR]", e)

    if SHOW_WINDOW:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
