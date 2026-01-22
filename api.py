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
from settings import AUTH_USER, FLASK_HOST, FLASK_PORT, HEARTBEAT_URL


app = Flask(__name__)
engine = FaceEngine()
engine.start_watcher()

def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

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


if __name__ == "__main__":
    t = threading.Thread(target=heartbeat_job, daemon=True)
    t.start()
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)