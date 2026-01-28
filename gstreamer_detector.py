import cv2
import time
import base64
import numpy as np
from face_engine import FaceEngine
from ws_client import WSClient
from settings import (
    VIDEO_SOURCE,
    COOLDOWN,
    TH_ACCEPT,
    WS_ENABLE,
    WS_URL,
    WS_JPEG_QUALITY,
)

# ======================
# WS
# ======================
ws = WSClient(WS_URL) if WS_ENABLE and WS_URL else None
print("[INFO] WS:", WS_ENABLE, WS_URL)

# ======================
# FACE ENGINE
# ======================
engine = FaceEngine()
engine.start_watcher()

# ======================
# GSTREAMER PIPELINE
# ======================
def gst_pipeline(rtsp_url):
    return (
        f"rtspsrc location={rtsp_url} latency=0 protocols=udp ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

print("[INFO] Using GStreamer RTSP")
cap = cv2.VideoCapture(gst_pipeline(VIDEO_SOURCE), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream with GStreamer")

# ======================
# HELPERS
# ======================
def encode_frame(frame):
    ok, buf = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, WS_JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

# ======================
# STATE
# ======================
last_sent = {}
TARGET_FPS = 3
last_proc = 0

print("[INFO] Realtime face daemon running (GStreamer)")

# ======================
# MAIN LOOP
# ======================
while True:
    now = time.time()

    if now - last_proc < 1 / TARGET_FPS:
        time.sleep(0.005)
        continue
    last_proc = now

    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    # Optional resize (VERY recommended)
    frame = cv2.resize(frame, (640, 360))

    results = engine.recognize(frame)

    ws_payload = {
        "type": "face_event",
        "name": None,
        "distance": None,
        "status": "NO_FACE",
        "box": None,
        "timestamp": int(now),
        "frame": None,
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

        ws_payload.update({
            "name": name,
            "distance": round(dist, 4),
            "status": status,
            "box": [x1, y1, x2, y2],
        })

        if status == "PASS":
            ws_payload["frame"] = encode_frame(frame)

            if now - last_sent.get(name, 0) >= COOLDOWN:
                last_sent[name] = now
                # TODO: webhook trigger

    if ws:
        ws.send(ws_payload)
