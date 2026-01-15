# 🔍 Face Recognition Service (Flask + OpenCV + WebSocket)

A **lightweight real-time face recognition system** built with **Python**, **Flask**, **OpenCV**, and a custom `FaceEngine`, now enhanced with **WebSocket streaming** for realtime frontend consumption.

This project supports:
- Single face registration
- Batch registration using ZIP
- Face unregistration
- Listing registered faces
- Realtime face recognition from camera
- Webhook event on successful recognition
- Realtime WebSocket face event + frame streaming

Designed for:
- Access control systems
- Attendance systems
- Gate / security automation
- Smart camera pipelines
- Live monitoring dashboard (frontend)

---

## 🧠 Architecture Overview

```
[ Camera / CCTV ]
        ↓
Realtime Detector (OpenCV)
        ↓
FaceEngine (detect + encode + recognize)
        ↓
 ├─ Webhook (PASS event)
 └─ WebSocket Stream (frame + face event)
        ↓
 Frontend / Dashboard
```

Components:
- Flask API Server – face management
- Realtime Face Daemon – camera loop
- WebSocket Server – broadcast realtime events
- FaceEngine – core recognition logic

---

## 📦 Requirements

### System
- Python 3.8+
- Linux / macOS / Windows
- USB Camera / RTSP CCTV

### Python Dependencies

```bash
pip install -r requirements.txt
```

> Additional dependencies may be required depending on `FaceEngine` (e.g. `dlib`, `face_recognition`).

---

## 📁 Project Structure

```
.
├── api.py                 # Flask REST API
├── detector.py            # Realtime face recognition daemon
├── ws_server.py           # WebSocket broadcast server
├── ws_client.py           # WebSocket client helper
├── face_engine.py         # Core face engine
├── settings.py            # Global configuration
├── faces/                 # Temporary extracted ZIP images
├── face_db.pkl            # Face database (encodings)
└── README.md
```

---

## 🚀 Running the API Server

```bash
python api.py
```

Server runs on:

```
http://0.0.0.0:5000
```

---

## 🔗 API Endpoints

### Register Single Face

POST `/register`

Form Data:
- `name` (string)
- `image` (file)

```bash
curl -X POST http://localhost:5000/register \
  -F "name=John" \
  -F "image=@john.jpg"
```

---

### Unregister Face

POST `/unregister`

```bash
curl -X POST http://localhost:5000/unregister \
  -F "name=John"
```

---

### List Registered Faces

GET `/faces`

```json
["katya", "zero", "ema"]
```

---

### Register Faces Using ZIP

POST `/register-faces`

ZIP structure:
```
faces.zip
 ├── katya.jpg
 ├── zero.jpg
 └── ema.jpg
```

```bash
curl -X POST http://localhost:5000/register-faces \
  -F "zip=@faces.zip"
```

---

## 🎥 Realtime Face Recognition Daemon

File: `detector.py`

Features:
- Realtime camera capture
- Face detection + recognition
- Bounding box & label rendering
- Cooldown per identity
- Webhook trigger on PASS
- WebSocket streaming (frame + metadata)

### Run

```bash
python detector.py
```

---

## 📡 WebSocket Server

File: `ws_server.py`

Start the WebSocket broadcast server:

```bash
python ws_server.py
```

Server URL:

```
ws://0.0.0.0:3001/ws/face-stream
```

Behavior:
- Multiple clients supported
- Incoming message is broadcast to all clients
- Designed for frontend dashboards

---

## 🔌 WebSocket Payload (Realtime)

Sent every frame from detector:

```json
{
  "type": "face_event",
  "name": "katya",
  "distance": 0.4123,
  "status": "PASS",
  "box": [120, 80, 240, 300],
  "timestamp": 1736400000,
  "frame": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

### Status Values
- `NO_FACE` – no face detected
- `PASS` – recognized & accepted
- `REJECT` – recognized but below threshold

Notes:
- `frame` is base64 JPEG
- Bounding box format: `[x1, y1, x2, y2]`
- Currently sends first detected face only
- Can be extended to multi-face array

---

## 📡 Webhook Payload (PASS only)

Triggered when:
- status == PASS
- Cooldown expired

```json
{
  "name": "katya",
  "distance": 0.4123,
  "timestamp": 1736400000
}
```

---

## ⚙️ Configuration (`settings.py`)

```python
VIDEO_SOURCE = 0
SCALE = 0.5
COOLDOWN = 5
TH_ACCEPT = 0.45

SHOW_WINDOW = True
WEBHOOK_URL = "http://localhost:3000/face-event"

WS_ENABLE = true
WS_URL = "ws://localhost:3001/ws/face-stream"
WS_JPEG_QUALITY = 70
```

---

## ⚡ Performance Tips

- Disable window rendering on headless devices
- Lower `WS_JPEG_QUALITY` to save bandwidth
- Avoid resizing every loop if possible
- `CAP_PROP_BUFFERSIZE = 1` minimizes latency
- CPU-friendly, tested on:
  - Jetson Nano
  - Orange Pi
  - Raspberry Pi

Tested with ~1K registered faces on Jetson Nano.

---

## 🧩 FaceEngine Notes

`FaceEngine` handles:
- Face detection
- Face encoding
- Distance matching
- Database persistence
- Background watcher thread

Ensure `face_engine.py` matches your hardware backend (HOG / CNN).

---

## 🛡️ Security Notes

- Do not expose API publicly without authentication
- Add API key / JWT for production
- Limit ZIP upload size
- Secure WebSocket endpoint if exposed externally
