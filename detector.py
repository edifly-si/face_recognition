import cv2
import time
import base64
from face_engine import FaceEngine
from ws_client import WSClient
from settings import (
    VIDEO_SOURCE,
    COOLDOWN,
    SHOW_WINDOW,
    TH_ACCEPT,
    WS_ENABLE,
    WS_URL,
    WS_JPEG_QUALITY,
)

# ======================
# WS INIT
# ======================
ws = None
if WS_ENABLE and WS_URL:
    ws = WSClient(WS_URL)

print("[INFO] WS:", WS_ENABLE, WS_URL)

# ======================
# FACE ENGINE
# ======================
engine = FaceEngine()
engine.start_watcher()

# ======================
# VIDEO CAPTURE
# ======================
print("[INFO] Video source:", VIDEO_SOURCE)

if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.startswith("rtsp"):
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    print("[INFO] RTSP mode (low latency)")
else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print("[INFO] Local camera mode")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ======================
# HELPERS
# ======================
def encode_frame(frame):
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, WS_JPEG_QUALITY],
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def read_latest(cap, skip=15):
    """
    Drop old buffered frames (VERY IMPORTANT FOR RTSP)
    """
    for _ in range(skip):
        cap.grab()
    return cap.retrieve()


# ======================
# STATE
# ======================
last_sent = {}
TARGET_FPS = 5
last_proc = 0

print("[INFO] Realtime face daemon running")

# ======================
# MAIN LOOP
# ======================
while True:
    now = time.time()

    # FPS limiter (prevents RTSP lag)
    if now - last_proc < 1 / TARGET_FPS:
        time.sleep(0.005)
        continue
    last_proc = now

    ret, frame = read_latest(cap)
    if not ret:
        time.sleep(0.1)
        continue

    results = engine.recognize(frame)

    ws_payload = {
        "type": "face_event",
        "name": None,
        "distance": None,
        "status": "NO_FACE",
        "box": None,
        "timestamp": int(now),
        "frame": None,  # send frame only when needed
    }

    for r in results:
        name = r["name"]
        dist = r["distance"]
        x1, y1, x2, y2 = r["box"]

        if dist > TH_ACCEPT:
            color = (0, 255, 0)
            label = "PASS"
            status = "PASS"
            name = None
        else:
            color = (0, 0, 255)
            label = f"{name} ({dist:.3f})"
            status = "REJECT"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        print(f"[GATE] {status} | {label}")

        ws_payload.update(
            {
                "name": name,
                "distance": round(dist, 4),
                "status": status,
                "box": [x1, y1, x2, y2],
            }
        )

        # only send image on PASS
        if status == "PASS":
            ws_payload["frame"] = encode_frame(frame)

            if now - last_sent.get(name, 0) >= COOLDOWN:
                last_sent[name] = now
                # TODO: webhook trigger

    if ws:
        ws.send(ws_payload)

    if SHOW_WINDOW:
        cv2.putText(
            frame,
            time.strftime("%H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
