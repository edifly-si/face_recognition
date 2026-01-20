import dlib
import cv2
import numpy as np
import pickle
import os
import threading
import time
from settings import (
    DB_PATH, FACES_DIR, THRESHOLD,
    SHAPE_MODEL, FACE_MODEL
)

os.makedirs(FACES_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(SHAPE_MODEL)
facerec = dlib.face_recognition_model_v1(FACE_MODEL)

class FaceEngine:
    def __init__(self):
        self.db = {}
        self.db_mtime = 0
        self.lock = threading.Lock()
        self.load_db(force=True)
        
    def find_similar(self, desc):
        for name, db_desc in self.db.items():
            dist = np.linalg.norm(desc - db_desc)
            if dist < THRESHOLD:
                return name, dist
        return None, None


    # =====================
    # DB LOAD
    # =====================
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

    # =====================
    # WATCHER THREAD
    # =====================
    def start_watcher(self, interval=1):
        def watch():
            while True:
                self.load_db()
                time.sleep(interval)

        threading.Thread(target=watch, daemon=True).start()

    # =====================
    # SAVE DB (ATOMIC)
    # =====================
    def _save_db(self):
        tmp = DB_PATH + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.db, f)
        os.replace(tmp, DB_PATH)

    # =====================
    # REGISTER (single image)
    # =====================
    def register(self, name, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        if len(dets) != 1:
            return False, "Face not detected or multiple faces found"

        shape = sp(gray, dets[0])
        desc = np.array(facerec.compute_face_descriptor(frame, shape))

        with self.lock:
            old_name, dist = self.find_similar(desc)

            if old_name:
                print(f"[REPLACE] {old_name} -> {name} (dist={dist:.4f})")
                del self.db[old_name]

            self.db[name] = desc
            self._save_db()

        if old_name:
            return True, f"Face replace from {old_name} to {name}"
        else:
            return True, f"Face {name} registered successfully"


    # =====================
    # UNREGISTER
    # =====================
    def unregister(self, name):
        with self.lock:
            if name not in self.db:
                return False, "Wajah tidak ditemukan"

            del self.db[name]
            self._save_db()

        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(FACES_DIR, name + ext)
            if os.path.exists(p):
                os.remove(p)

        return True, f"Wajah {name} dihapus"

    # =====================
    # RECOGNIZE
    # =====================
    def recognize(self, frame):
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        with self.lock:
            db_snapshot = self.db.copy()

        for d in dets:
            shape = sp(gray, d)
            cur_desc = np.array(
                facerec.compute_face_descriptor(frame, shape)
            )

            best_name = "UNKNOWN"
            best_dist = 999.0

            for name, db_desc in db_snapshot.items():
                dist = np.linalg.norm(cur_desc - db_desc)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            results.append({
                "name": best_name if best_dist < THRESHOLD else "UNKNOWN",
                "distance": float(best_dist),
                "box": [d.left(), d.top(), d.right(), d.bottom()]
            })

        return results

    # =====================
    # REGISTER FROM FOLDER
    # =====================
    def register_from_folder(self, folder):
        if not os.path.isdir(folder):
            return False, "Folder tidak ada"

        success, failed = 0, []

        for f in os.listdir(folder):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            name = os.path.splitext(f)[0]
            path = os.path.join(folder, f)

            img = cv2.imread(path)
            if img is None:
                failed.append(f)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray)
            if len(dets) != 1:
                failed.append(f)
                continue

            shape = sp(gray, dets[0])
            desc = np.array(
                facerec.compute_face_descriptor(img, shape)
            )

            with self.lock:
                self.db[name] = desc

            success += 1

        if success:
            self._save_db()

        return True, {"registered": success, "failed": failed}
