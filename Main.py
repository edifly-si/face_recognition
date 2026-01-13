import os
import cv2
import dlib
import numpy as np
import time

# =====================
# CONFIG
# =====================
CAMERA_INDEX = 0
FACE_DIR = "Face_Directory"
THRESHOLD = 0.6

# =====================
# INIT DLIB
# =====================
print("[INIT] Loading dlib models...")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)

# =====================
# LOAD DATASET
# =====================
known_faces = {}

print("[DATASET] Loading faces...")
for file in os.listdir(FACE_DIR):
    if not file.lower().endswith((".jpg", ".png")):
        continue

    label = os.path.splitext(file)[0]
    img = cv2.imread(os.path.join(FACE_DIR, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = detector(gray)
    if len(dets) == 0:
        print(f"[SKIP] No face detected in {file}")
        continue

    shape = sp(gray, dets[0])
    desc = np.array(facerec.compute_face_descriptor(img, shape))
    known_faces[label] = desc
    print(f"[OK] Loaded {label}")

if not known_faces:
    print("[FATAL] No faces loaded")
    exit(1)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[FATAL] Camera not opened")
    exit(1)

print("[RUNNING] Press Q to exit\n")

# =====================
# LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame grab failed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)

    if not dets:
        print("[INFO] No face detected")
        time.sleep(0.3)
        continue

    for d in dets:
        shape = sp(gray, d)
        cur_desc = np.array(facerec.compute_face_descriptor(frame, shape))

        distances = {}
        for name, known_desc in known_faces.items():
            dist = np.linalg.norm(cur_desc - known_desc)
            distances[name] = dist

        best_name, best_dist = min(distances.items(), key=lambda x: x[1])

        if best_dist < THRESHOLD:
            print(f"[MATCH] {best_name} | distance={best_dist:.4f}")
        else:
            print(f"[UNKNOWN] closest={best_name} | distance={best_dist:.4f}")

    time.sleep(0.2)

# =====================
# CLEANUP
# =====================
    cap.release()
    cv2.destroyAllWindows()
