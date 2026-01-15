import cv2
import time
from face_engine import FaceEngine
import base64
from ws_client import WSClient
from settings import WS_URL, WS_JPEG_QUALITY

from settings import (
    VIDEO_SOURCE, COOLDOWN,
    SHOW_WINDOW, TH_ACCEPT, WS_ENABLE
)

ws = None
if WS_ENABLE:
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

    results = engine.recognize(frame)
    now = time.time()

    ws_payload = {
        "type": "face_event",
        "name": None,
        "distance": None,
        "status": "NO_FACE",
        "box": None,
        "timestamp": int(now),
        "frame": encode_frame(frame) if ws else None
    }

    for r in results:
        name = r["name"]
        dist = r["distance"]
        x1, y1, x2, y2 = r["box"]

        if dist > TH_ACCEPT:
            color = (0, 255, 0)
            label =  "PASS"
            status = "PASS"
            name = None
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

        ws_payload.update({
            "name": name,
            "distance": round(dist, 4),
            "status": status,
            "box": [x1, y1, x2, y2]
        })

        if status == "PASS" and now - last_sent.get(name, 0) >= COOLDOWN:
            last_sent[name] = now
            # WEBHOOK trigger

    if ws:
        ws.send(ws_payload)

    if SHOW_WINDOW:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
