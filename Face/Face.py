# face_module.py
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from openpyxl import Workbook, load_workbook
from insightface.app import FaceAnalysis
from collections import defaultdict
import threading

from PySide6.QtWidgets import QApplication, QDialog, QGraphicsScene
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtGui import QImage, QPixmap

from ui_main_window import Ui_Start_2


# PyInstaller path fix
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


model_path = resource_path("insightface/models")
os.environ["INSIGHTFACE_HOME"] = model_path

# Initialize InsightFace
face_recognition = FaceAnalysis(allowed_modules=['detection', 'recognition'], root=model_path)
face_recognition.prepare(ctx_id=-1, det_size=(640, 640))

# Load known face encodings
known_face_encodings = defaultdict(list)
known_faces_dir = 'known_faces'
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path)
            faces = face_recognition.get(img)
            if faces:
                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                known_face_encodings[person_name].append(embedding)

excel_filename = 'attendance.xlsx'
try:
    wb = load_workbook(excel_filename)
    ws = wb.active
except FileNotFoundError:
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Date", "Time"])


class VideoThread(QThread):
    frame_data = Signal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        already_marked = set()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            #frame = cv2.flip(frame, 0)
            faces = face_recognition.get(frame)

            for face in faces:
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)
                best_match = "Unknown"
                best_distance = float('inf')

                for name, embeddings in known_face_encodings.items():
                    distances = np.linalg.norm(np.array(embeddings) - embedding, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_match = name

                threshold = 4
                name = best_match if best_distance < threshold else "Unknown"
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (125, 100, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

                if name != "Unknown" and name not in already_marked:
                    now = datetime.now()
                    ws.append([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    already_marked.add(name)

            # Convert frame (BGR) to QImage (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_data.emit(qt_image)

        cap.release()
        wb.save(excel_filename)

    def stop(self):
        self.running = False
        self.wait()


class MainApp(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.ui = Ui_Start_2()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        self.pixmap_item = None

        self.video_thread = VideoThread()
        self.video_thread.frame_data.connect(self.update_frame)

        self.ui.Start.clicked.connect(self.start_attendance)
        self.ui.pushButton_2.clicked.connect(self.stop_attendance)

    @Slot(QImage) 
    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.ui.graphicsView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        else:
            self.pixmap_item.setPixmap(pixmap)
            self.ui.graphicsView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def start_attendance(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()

    def stop_attendance(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
