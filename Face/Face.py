import os
import cv2
import re
import numpy as np
from datetime import datetime
from openpyxl import Workbook, load_workbook
from insightface.app import FaceAnalysis
from collections import defaultdict


app = FaceAnalysis(name='buffalo_l')  # You can use 'antelopev2' if needed
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, -1 for CPU

# Load known faces and their embeddings
known_face_encodings = defaultdict(list)
known_face_names = []



known_faces_dir = 'known_faces'
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path)
            faces = app.get(img)
            if faces:
                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                known_face_encodings[person_name].append(embedding)
            else:
                print(f"[WARNING] No face found in {filename}")

# Flattened for faster access if needed
known_face_names = list(known_face_encodings.keys())
# -------------------- Attendance Setup --------------------
excel_filename = 'attendance.xlsx'
try:
    wb = load_workbook(excel_filename)
    ws = wb.active
except FileNotFoundError:
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Date", "Time"])

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)
already_marked = set()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    #frame = cv2.flip(frame,0)

    faces = app.get(frame)
    for face in faces:
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)  # Normalize the input embedding

        best_match = "Unknown"
        best_distance = float('inf')

        for name, embeddings in known_face_encodings.items():
            distances = np.linalg.norm(np.array(embeddings) - embedding, axis=1)
            min_dist = np.min(distances)

            if min_dist < best_distance:
                best_distance = min_dist
                best_match = name

        threshold = 0.78  # You can tune this
        name = best_match if best_distance < threshold else "Unknown"

        # Draw box and name
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (125, 100, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        print(f"[DEBUG] Match for {name} with distance {best_distance}")

        if name != "Unknown" and name not in already_marked:
            now = datetime.now()
            ws.append([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
            already_marked.add(name)
            print(f"[INFO] Marked present: {name}")

    
    cv2.imshow("InsightFace Attendance",frame )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------
wb.save(excel_filename)
cap.release()
cv2.destroyAllWindows()