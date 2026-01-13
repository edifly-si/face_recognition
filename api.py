from flask import Flask, request, jsonify
import cv2
import numpy as np
from face_engine import FaceEngine

app = Flask(__name__)
engine = FaceEngine()
engine.start_watcher()

def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# =====================
# REGISTER
# =====================
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"error": "name & image required"}), 400

    frame = read_image(file)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    ok, msg = engine.register(name, frame)
    return jsonify({"success": ok, "message": msg})

# =====================
# UNREGISTER
# =====================
@app.route("/unregister", methods=["POST"])
def unregister():
    name = request.form.get("name")

    if not name:
        return jsonify({"error": "name required"}), 400

    ok, msg = engine.unregister(name)
    return jsonify({"success": ok, "message": msg})

# =====================
# LIST FACES
# =====================
@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify(list(engine.db.keys()))

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False
    )
