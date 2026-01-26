from flask import Flask, request, jsonify
import threading
import time
import requests
import cv2
import numpy as np
import tempfile
import os
from zipfile import ZipFile
from api_helper import require_basic_auth
from face_engine import FaceEngine
import base64
from requests.auth import HTTPBasicAuth
from settings import AUTH_USER, FLASK_HOST, FLASK_PORT, HEARTBEAT_URL, SYNC_BASE_URL, SYNC_USER, SYNC_PASSWORD, FILE_SYNC_URL

SYNC_INTERVAL = 30 * 60
SYNC_SINCE_FILE = "sync_since.txt"
app = Flask(__name__)
engine = FaceEngine()
engine.start_watcher()

def node_auth():
    return HTTPBasicAuth(SYNC_USER, SYNC_PASSWORD)

def base64_to_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def load_since():
    if os.path.exists(SYNC_SINCE_FILE):
        with open(SYNC_SINCE_FILE, "r") as f:
            value = f.read().strip()
            return value or None
    return None

def save_since(value):
    if not value:
        return
    with open(SYNC_SINCE_FILE, "w") as f:
        f.write(str(value))


# =====================
# REGISTER SINGLE
# =====================
@app.route("/register", methods=["POST"])
@require_basic_auth
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return {"error": "name & image required"}, 400

    frame = read_image(file)
    ok, msg = engine.register(name, frame)
    return {"success": ok, "message": msg}

# =====================
# UNREGISTER
# =====================
@app.route("/unregister", methods=["POST"])
@require_basic_auth
def unregister():
    name = request.form.get("name")
    if not name:
        return {"error": "name required"}, 400

    ok, msg = engine.unregister(name)
    return {"success": ok, "message": msg}

# =====================
# LIST
# =====================
@app.route("/faces", methods=["GET"])
@require_basic_auth
def faces():
    return jsonify(list(engine.db.keys()))

# =====================
# REGISTER ZIP
# =====================
@app.route("/register-faces", methods=["POST"])
@require_basic_auth
def register_zip():
    if "zip" not in request.files:
        return {"error": "zip missing"}, 400

    zip_file = request.files["zip"]
    tmp = tempfile.gettempdir()
    zip_path = os.path.join(tmp, zip_file.filename)
    zip_file.save(zip_path)

    with ZipFile(zip_path, "r") as z:
        z.extractall("faces")

    ok, result = engine.register_from_folder("faces")
    return {"success": ok, "result": result}

def heartbeat_job():
    time.sleep(3)

    url = f"{HEARTBEAT_URL}/{AUTH_USER}"

    while True:
        try:
            resp = requests.get(
                url,
                timeout=5,
                params={"name": "heartbeat"},
            )
            print("[HEARTBEAT]", resp.status_code)
        except Exception as e:
            print(e)
            
            print("[HEARTBEAT ERROR]", e)

        time.sleep(30)

def syncFace():
    time.sleep(10)

    since = load_since()
    print(f"[SYNC_BASE_URL] face sync started (since={since})")

    while True:
        try:
            params = {}
            if since:
                params["since"] = since

            resp = requests.get(
                f"{SYNC_BASE_URL}/blacklist",
                auth=node_auth(),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()

            raw = resp.json()

            if isinstance(raw, dict):
                if "data" in raw:
                    items = raw["data"]
                elif "items" in raw:
                    items = raw["items"]
                else:
                    print("[SYNC_BASE_URL] unknown response shape:", raw)
                    items = []
            elif isinstance(raw, list):
                items = raw
            else:
                print("[SYNC_BASE_URL] invalid response type:", type(raw))
                items = []

            for item in items:
                name = item.get("spectra_id")
                image_name = item.get("image_name")

                if not name or not image_name:
                    continue

                img_resp = requests.get(
                    f"{FILE_SYNC_URL}/{image_name}",
                    auth=node_auth(),
                    timeout=15,
                )
                img_resp.raise_for_status()

                raw = img_resp.json()

                if raw.get("error") != 0:
                    print("[FILE_SYNC_URL] image error:", raw)
                    continue

                data = raw.get("data")
                if not data or "base64" not in data:
                    print("[FILE_SYNC_URL] invalid image payload:", raw)
                    continue

                frame = base64_to_image(data["base64"])

                ok, msg = engine.register(name, frame)
                print(f"[SYNC_BASE_URL] {name}: {msg}")

                since = item.get("updatedAt") or item.get("createdAt")
                save_since(since)

        except Exception as e:
            print("[SYNC_BASE_URL ERROR]", e)

        print("[SYNC_BASE_URL] sleeping 30 minutes...\n")
        time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    t = threading.Thread(target=heartbeat_job, daemon=True)
    threading.Thread(target=syncFace, daemon=True).start()

    t.start()
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)