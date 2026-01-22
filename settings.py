import os
from dotenv import load_dotenv

load_dotenv()

def _cast(val):
    if val is None:
        return None
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except:
        try:
            return float(val)
        except:
            return val

DB_PATH = os.getenv("DB_PATH", "face_db.pkl")
FACES_DIR = os.getenv("FACES_DIR", "faces")

SHAPE_MODEL = os.getenv("SHAPE_MODEL")
FACE_MODEL = os.getenv("FACE_MODEL")

THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
TH_ACCEPT = float(os.getenv("TH_ACCEPT", 0.4))

VIDEO_SOURCE = _cast(os.getenv("VIDEO_SOURCE", "0"))
SCALE = float(os.getenv("SCALE", 0.5))
SHOW_WINDOW = _cast(os.getenv("SHOW_WINDOW", "true"))
COOLDOWN = int(os.getenv("COOLDOWN", 5))

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
AUTH_USER = os.getenv("AUTH_USER")
AUTH_PASS = os.getenv("AUTH_PASS")

WS_ENABLE = _cast(os.getenv("WS_ENABLE", "false"))
WS_URL = os.getenv("WS_URL")
WS_JPEG_QUALITY = int(os.getenv("WS_JPEG_QUALITY", 70))

HEARTBEAT_URL = os.getenv("HEARTBEAT_URL")


