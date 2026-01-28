import cv2
import time
import threading
import base64
from face_engine import FaceEngine
from ws_client import WSClient
from settings import (
    VIDEO_SOURCE,
    WS_URL,
    WS_ENABLE,
    WS_JPEG_QUALITY,
    COOLDOWN,
    TH_ACCEPT,
)

class FastRTSP:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            # buang frame lama
            for _ in range(5):
                self.cap.grab()

            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.frame

    def release(self):
        self.running = False
        self.cap.release()

def encode_frame(frame):
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, WS_JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

ws = None
if WS_ENABLE and WS_URL:
    ws = WSClient(WS_URL)

print("[INFO] WS:", WS_ENABLE, WS_URL)


engine = FaceEngine()
engine.start_watcher()


print("[INFO] Video source:", VIDEO_SOURCE)

if VIDEO_SOURCE.startswith("rtsp://"):
    print("[INFO] RTSP mode (OpenCV FFmpeg)")
    cam = FastRTSP(VIDEO_SOURCE)
else:
    cam = cv2.VideoCapture(VIDEO_SOURCE)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_sent = {}
print("[INFO] Realtime face daemon running")

# =========================
# MAIN LOOP
# =========================
while True:
    frame = cam.read() if isinstance(cam, FastRTSP) else cam.read()[1]

    if frame is None:
        time.sleep(0.02)
        continue

    now = time.time()
    results = engine.recognize(frame)

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
            status = "PASS"
            name = None
        else:
            status = "REJECT"

        print(f"[GATE] {status} | {name} | {dist:.3f}")

        ws_payload.update({
            "name": name,
            "distance": round(dist, 4),
            "status": status,
            "box": [x1, y1, x2, y2]
        })

        if status == "PASS":
            if now - last_sent.get(name, 0) >= COOLDOWN:
                last_sent[name] = now

    if ws:
        ws.send(ws_payload)

    # kontrol CPU
    time.sleep(0.03)

