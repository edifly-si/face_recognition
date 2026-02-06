import cv2
import time
import threading
import base64
import statistics
from collections import defaultdict, deque
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

# =========================
# STRICT CONFIG
# =========================
CONFIRM_FRAMES = 5
HISTORY_SIZE = 5
MAX_STD = 0.03

history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
confirmed = {}
last_reject_sent = {}

# =========================
# FAST RTSP
# =========================
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

# =========================
# FRAME ENCODER
# =========================
def encode_frame(frame):
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, WS_JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

# =========================
# INIT
# =========================
ws = WSClient(WS_URL) if WS_ENABLE and WS_URL else None
print("[INFO] WS:", WS_ENABLE, WS_URL)

engine = FaceEngine()
engine.start_watcher()

print("[INFO] Video source:", VIDEO_SOURCE)

if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.startswith("rtsp://"):
    print("[INFO] RTSP mode")
    cam = FastRTSP(VIDEO_SOURCE)
else:
    print("[INFO] Local camera mode")
    cam = cv2.VideoCapture(VIDEO_SOURCE)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    status = "NO_FACE"
    name = None
    distance = None
    box = None

    if results:
        status = "PASS"

    for r in results:
        r_name = r["name"]
        r_dist = r["distance"]
        r_box = r["box"]

        if r_name == "UNKNOWN":
            continue

        history[r_name].append(r_dist)

        if len(history[r_name]) < CONFIRM_FRAMES:
            continue

        avg = sum(history[r_name]) / len(history[r_name])
        std = statistics.pstdev(history[r_name])

        if avg < TH_ACCEPT and std < MAX_STD:
            status = "REJECT"
            name = r_name
            distance = round(avg, 4)
            box = r_box

            last = last_reject_sent.get(r_name, 0)
            if now - last >= COOLDOWN:
                last_reject_sent[r_name] = now
                print(f"[REJECT] {r_name} avg={avg:.3f} std={std:.3f}")
            break

    # =========================
    # WS PAYLOAD (KIRIM TERUS)
    # =========================
    payload = {
        "type": "face_event",
        "status": status,
        "name": name,
        "distance": distance,
        "box": box,
        "timestamp": int(now),
        "frame": encode_frame(frame) if ws else None
    }

    if ws:
        print("[WS] Sending payload:", payload["status"], payload["name"], payload["distance"])
        ws.send(payload)

    time.sleep(0.03)
