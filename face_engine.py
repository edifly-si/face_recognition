import dlib
import cv2
import numpy as np
import pickle
import os
import threading
import time
import math
from settings import (
    DB_PATH, FACES_DIR, TH_ACCEPT, THRESHOLD,
    SHAPE_MODEL, FACE_MODEL, FACE_DETECTOR_MODE, CNN_MODEL
)

os.makedirs(FACES_DIR, exist_ok=True)

# =========================
# DETECTOR
# =========================
if FACE_DETECTOR_MODE == "cuda":
    print("[FACE] Detector mode: CUDA CNN")
    detector = dlib.cnn_face_detection_model_v1(CNN_MODEL)
else:
    print("[FACE] Detector mode: CPU HOG")
    detector = dlib.get_frontal_face_detector()

sp = dlib.shape_predictor(SHAPE_MODEL)
facerec = dlib.face_recognition_model_v1(FACE_MODEL)

# =========================
# POSE FILTER
# =========================
def is_face_straight(shape, max_roll=10, max_yaw=0.15):
    lx = (shape.part(36).x + shape.part(39).x) / 2
    ly = (shape.part(36).y + shape.part(39).y) / 2
    rx = (shape.part(42).x + shape.part(45).x) / 2
    ry = (shape.part(42).y + shape.part(45).y) / 2

    roll = math.degrees(math.atan2(ry - ly, rx - lx))
    if abs(roll) > max_roll:
        return False

    nose_x = shape.part(30).x
    eye_center_x = (lx + rx) / 2
    face_width = abs(rx - lx)

    yaw = abs(nose_x - eye_center_x) / face_width
    if yaw > max_yaw:
        return False

    return True

# =========================
# FACE ENGINE
# =========================
class FaceEngine:
    def __init__(self):
        self.db = {}
        self.db_mtime = 0
        self.lock = threading.Lock()
        self.load_db(force=True)

    # =========================
    # DB
    # =========================
    def load_db(self, force=False):
        if not os.path.exists(DB_PATH):
            return

        mtime = os.path.getmtime(DB_PATH)
        if force or mtime != self.db_mtime:
            try:
                with open(DB_PATH, "rb") as f:
                    data = pickle.load(f)

                with self.lock:
                    self.db = data
                    self.db_mtime = mtime

                print(f"[DB] Reloaded ({len(self.db)} faces)")
            except Exception as e:
                print("[DB] Reload failed:", e)

    def start_watcher(self, interval=1):
        def watch():
            while True:
                self.load_db()
                time.sleep(interval)
        threading.Thread(target=watch, daemon=True).start()

    def _save_db(self):
        tmp = DB_PATH + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.db, f)
        os.replace(tmp, DB_PATH)

    # =========================
    # CORE UTILS
    # =========================
    def _get_face_chip(self, rgb, shape):
        """
        KUNCI AKURASI:
        align + crop ke 150x150
        """
        return dlib.get_face_chip(rgb, shape, size=150)

    def find_similar(self, desc):
        best_name = None
        best_dist = 999

        for name, db_desc in self.db.items():
            dist = np.linalg.norm(desc - db_desc)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist < TH_ACCEPT:
            return best_name, best_dist
        return None, None

    # =========================
    # REGISTER
    # =========================
    def register(self, name, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if FACE_DETECTOR_MODE == "cuda":
            dets = detector(rgb, 1)
            rects = [d.rect for d in dets]
        else:
            rects = detector(gray)

        if len(rects) != 1:
            return {"error": "Face not detected or multiple faces found"}, 400

        rect = rects[0]
        shape = sp(rgb if FACE_DETECTOR_MODE == "cuda" else gray, rect)

        if not is_face_straight(shape, max_roll=8, max_yaw=0.12):
            return {"error": "face position is not straight"}, 400

        face_chip = self._get_face_chip(rgb, shape)
        desc = np.array(
            facerec.compute_face_descriptor(face_chip),
            dtype=np.float32
        )

        with self.lock:
            old_name, dist = self.find_similar(desc)
            if old_name:
                del self.db[old_name]
            self.db[name] = desc
            self._save_db()

        return True, f"Face {name} registered"

    # =========================
    # RECOGNIZE
    # =========================
    def recognize(self, frame):
        results = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if FACE_DETECTOR_MODE == "cuda":
            dets = detector(rgb, 0)
            rects = [d.rect for d in dets]
        else:
            rects = detector(gray)

        with self.lock:
            db_snapshot = self.db.copy()

        for rect in rects:
            shape = sp(rgb if FACE_DETECTOR_MODE == "cuda" else gray, rect)

            if not is_face_straight(shape, max_roll=10, max_yaw=0.18):
                continue

            face_chip = self._get_face_chip(rgb, shape)
            desc = np.array(
                facerec.compute_face_descriptor(face_chip),
                dtype=np.float32
            )

            best_name = "UNKNOWN"
            best_dist = 999.0

            for name, db_desc in db_snapshot.items():
                dist = np.linalg.norm(desc - db_desc)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            results.append({
                "name": best_name if best_dist < THRESHOLD else "UNKNOWN",
                "distance": float(best_dist),
                "box": [
                    rect.left(),
                    rect.top(),
                    rect.right(),
                    rect.bottom()
                ]
            })

        return results
