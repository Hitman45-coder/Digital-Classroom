import os
import sys
import cv2
import numpy as np
from datetime import datetime
from openpyxl import Workbook, load_workbook
from insightface.app import FaceAnalysis
from collections import defaultdict
import threading

from PySide6.QtWidgets import QApplication, QDialog, QGraphicsScene, QFileDialog, QLabel
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QInputDialog, QMessageBox, QDialogButtonBox, QVBoxLayout
from ui_main_window import Ui_Start_2
import shutil


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

                threshold = 1
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

        self.ui.pushButton_4.clicked.connect(self.add_face)
        self.ui.pushButton_5.clicked.connect(self.remove_face)

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


    def add_face(self):
        method, ok = QInputDialog.getItem(
            self, "Select Method", "Add face by:",
            ["Capture from Webcam", "Upload from Disk"],
            0, False
        )
        if not ok:
            return

        name, ok = QInputDialog.getText(self, "Person's Name", "Enter person's name:")
        if not ok or not name.strip():
            return

        name = name.strip()
        folder_path = os.path.join("known_faces", name)
        os.makedirs(folder_path, exist_ok=True)

        # Webcam mode
        if method == "Capture from Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Webcam not accessible.")
                return

            QMessageBox.information(self, "Instruction", "Press 's' to capture, 'q' to cancel.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Capture Face - Press 's' to save", frame)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # Upload from disk mode
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if not file_path:
                return
            frame = cv2.imread(file_path)

        # Preview and confirmation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape
        bytes_per_line = 3 * width
        preview_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Photo")
        layout = QVBoxLayout(dialog)

        label = QLabel()
        label.setPixmap(QPixmap.fromImage(preview_img).scaled(400, 400, Qt.KeepAspectRatio))
        layout.addWidget(label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            save_path = os.path.join(folder_path, f"{count + 1}.jpg")
            cv2.imwrite(save_path, frame)
            QMessageBox.information(self, "Success", f"Image saved to {save_path}")
            self.refresh_known_encodings()
        else:
            QMessageBox.information(self, "Cancelled", "Face not saved.")

        self.refresh_known_encodings()


    def remove_face(self):
        name, ok = QInputDialog.getText(self, "Remove Face", "Enter person's name to delete:")
        if not ok or not name.strip():
            return

        name = name.strip()
        folder_path = os.path.join("known_faces", name)

        if not os.path.exists(folder_path):
            QMessageBox.warning(self, "Error", "Person not found.")
            return

        confirm = QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete all data for {name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            shutil.rmtree(folder_path)
            QMessageBox.information(self, "Deleted", f"All data for {name} deleted.")
            self.refresh_known_encodings()


    def refresh_known_encodings(self):
        global known_face_encodings
        known_face_encodings = defaultdict(list)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
