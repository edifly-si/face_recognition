# 🔍 Face Recognition Service (Flask + OpenCV)

A **lightweight real-time face recognition system** built with **Python**, **Flask**, **OpenCV**, and a custom `FaceEngine`.

This project supports:
- Single face registration
- Batch registration using ZIP
- Face unregistration
- Listing registered faces
- Realtime face recognition from camera
- Webhook event on successful recognition

Designed for:
- Access control systems
- Attendance systems
- Gate / security automation
- Smart camera pipelines

---

## 🧠 Architecture Overview

API Server (Flask)
→ FaceEngine (encoding + database)
→ Realtime Camera Daemon
→ Webhook Event Receiver

---

## 📦 Requirements

### System
- Python 3.8 or higher
- Linux / macOS / Windows
- USB Camera or CCTV stream

### Python Dependencies

```bash
pip install flask opencv-python numpy requests
```

> Additional dependencies may be required depending on the internal implementation of `FaceEngine` (e.g. dlib).

---

## 📁 Project Structure

```
.
├── api.py                 # Flask REST API
├── realtime_daemon.py     # Realtime face recognition loop
├── face_engine.py         # Core face engine
├── faces/                 # Temporary extracted ZIP images
├── Face_Directory/        # Face database (encodings)
└── README.md
```

---

## 🚀 Running the API Server

```bash
python api.py
```

Server will start on:

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

Example:
```bash
curl -X POST http://localhost:5000/register   -F "name=John"   -F "image=@john.jpg"
```

---

### Unregister Face

POST `/unregister`

Form Data:
- `name` (string)

```bash
curl -X POST http://localhost:5000/unregister   -F "name=John"
```

---

### List Registered Faces

GET `/faces`

Response:
```json
["John", "Alice", "Bob"]
```

---

### Register Faces Using ZIP

POST `/register-faces`

ZIP format:
```
faces.zip
├── John/
│   ├── 1.jpg
│   └── 2.jpg
├── Alice/
│   └── alice.png
```

```bash
curl -X POST http://localhost:5000/register-faces   -F "zip=@faces.zip"
```

---

## 🎥 Realtime Face Recognition Daemon

File: `realtime_daemon.py`

Features:
- Realtime camera capture
- Frame scaling for performance
- Face recognition loop
- Cooldown per recognized face
- Webhook integration

### Run

```bash
python realtime_daemon.py
```

### Configuration

```python
CAMERA_INDEX = 0
SCALE = 0.5
COOLDOWN = 5
WEBHOOK_URL = "http://localhost:3000/face-event"
```

---

## 📡 Webhook Payload

Sent when a known face is detected:

```json
{
  "name": "John",
  "distance": 0.4123,
  "timestamp": 1736400000
}
```

Cooldown prevents repeated events for the same face.

---

## ⚙️ Performance Tips

- Reduce `SCALE` for higher FPS
- `CAP_PROP_BUFFERSIZE = 1` minimizes latency
- Works well on CPU-only devices
- Suitable for Jetson Nano, Orange Pi, Raspberry Pi

---

## 🧩 FaceEngine Notes

`FaceEngine` handles:
- Face detection
- Face encoding
- Database management
- Background watcher thread

Ensure `face_engine.py` is present and compatible.

---

## 🛡️ Security Notes

- Do not expose API publicly without authentication
- Add API keys or JWT for production use
- Limit ZIP upload size

---

## 📜 License

MIT License

---

## ✨ Author

Built for real-time computer vision systems.
