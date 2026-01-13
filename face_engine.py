import dlib
import cv2
import numpy as np
import pickle
import os
import threading
import time

DB_PATH = "face_db.pkl"
THRESHOLD = 0.6

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)

class FaceEngine:
    def __init__(self):
        self.db = {}
        self.db_mtime = 0
        self.lock = threading.Lock()
        self.load_db(force=True)

    # =====================
    # LOAD DB
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
    # REGISTER
    # =====================
    def register(self, name, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray)

        if len(dets) != 1:
            return False, "Harus tepat 1 wajah"

        shape = sp(gray, dets[0])
        desc = np.array(facerec.compute_face_descriptor(frame, shape))

        with self.lock:
            self.db[name] = desc
            self._save_db()

        return True, f"Wajah {name} berhasil diregister"

    # =====================
    # UNREGISTER
    # =====================
    def unregister(self, name):
        with self.lock:
            if name not in self.db:
                return False, "Wajah tidak ditemukan"

            del self.db[name]
            self._save_db()

        return True, f"Wajah {name} berhasil dihapus"

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
