import os
import sys
import cv2
import numpy as np
from datetime import datetime
from openpyxl import Workbook, load_workbook
from insightface.app import FaceAnalysis
from collections import defaultdict
import threading

from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QLabel, QDialogButtonBox, 
    QVBoxLayout, QInputDialog, QMessageBox, QTableWidgetItem, QHeaderView, QHBoxLayout, QSizePolicy,
    QWidget
)
from PySide6.QtCore import Qt, Signal, QThread, Slot, QRect
from PySide6.QtGui import QImage, QPixmap, QFont
from ui_main_window import Ui_Start_2
import shutil
import smtplib
from email.message import EmailMessage
import json
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file

SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

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
os.environ["INSIGHTFACE_DOWNLOAD_PROGRESS"] = "False"
face_recognition = FaceAnalysis(allowed_modules=['detection', 'recognition'], root=model_path)
face_recognition.prepare(ctx_id=-1, det_size=(640, 640))  # Set ctx_id to 0 for GPU

# Load known face encodings
known_face_encodings = defaultdict(list)
known_faces_dir = 'known_faces'
os.makedirs(known_faces_dir, exist_ok=True)
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
    frame_data = Signal(int, QImage)
    attendance_updated = Signal(str)  # Signal to update GUI attendance dynamically

    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.running = False
        self.camera_id = camera_id

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # Use DSHOW for USB devices
        if not cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return
        already_marked = set()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

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
                    self.attendance_updated.emit(name)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_data.emit(self.camera_id, qt_image)

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

        # Apply a main layout to the dialog
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Wrap video area in a widget with a vertical layout for multiple feeds
        self.video_widget = QWidget()
        self.video_layout = QVBoxLayout(self.video_widget)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(0)  # No spacing between feeds
        main_layout.addWidget(self.video_widget, stretch=2)

        # Group buttons and Attendance into a widget with a vertical layout
        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)
        button_layout.addWidget(self.ui.Start_Button)
        button_layout.addWidget(self.ui.Stop_Button)
        button_layout.addWidget(self.ui.Mail_Button)
        button_layout.addWidget(self.ui.Add_Button)
        button_layout.addWidget(self.ui.Remove_Button)
        button_layout.addWidget(self.ui.Attendance)
        button_layout.addStretch()  # Push buttons to the top
        button_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(button_panel, stretch=1)

        # Initialize present_today
        self.present_today = set()

        # Configure QTableWidget
        self.ui.Attendance.setColumnCount(1)
        self.ui.Attendance.setHorizontalHeaderLabels(["Present Today"])
        self.ui.Attendance.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ui.Attendance.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.camera_threads = {}  # Camera_id: VideoThread
        self.camera_views = {}   # camera_id: QLabel

        # Connect buttons
        self.ui.Start_Button.clicked.connect(self.start_attendance)
        self.ui.Stop_Button.clicked.connect(self.stop_attendance)
        self.ui.Add_Button.clicked.connect(self.add_face)
        self.ui.Remove_Button.clicked.connect(self.remove_face)
        self.ui.Mail_Button.clicked.connect(self.send_absentee_emails)

        # Initialize the attendance list
        self.update_today_present_list()

    def resizeEvent(self, event):
        """Handle window resize to adjust video feed sizes."""
        super().resizeEvent(event)
        for cam_id, label in self.camera_views.items():
            self._adjust_label_size(label)

    def _adjust_label_size(self, label):
        """Adjust label size based on parent widget and number of cameras."""
        if not self.video_widget or not self.video_layout.count():
            return
        total_height = self.video_widget.height()
        if len(self.camera_views) == 1:
            label.setMaximumSize(self.video_widget.width(), total_height)
        else:
            label.setMaximumSize(self.video_widget.width(), total_height // len(self.camera_views))

    @Slot(int, QImage)
    def update_frame(self, cam_id, image):
        pixmap = QPixmap.fromImage(image)
        if cam_id in self.camera_views:
            label = self.camera_views[cam_id]
            # Scale pixmap to fit the label's maximum allowed size while maintaining aspect ratio
            max_size = label.maximumSize()
            scaled_pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)

    def detect_connected_cameras(self, max_cams=2):
        available_cams = []
        for i in range(max_cams):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW for better USB support
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cams.append(i)
            cap.release()
        print(f"Detected cameras: {available_cams}")  # Debug print
        return available_cams

    def start_attendance(self):
        if self.camera_threads:
            return  # Already running

        cameras = self.detect_connected_cameras()
        
        if not cameras:
            QMessageBox.warning(self, "No Cameras", "No cameras detected")
            return

        # Clear existing widgets in the video_layout
        for i in reversed(range(self.video_layout.count())):
            widget = self.video_layout.itemAt(i).widget()
            if widget:
                self.video_layout.removeWidget(widget)
                widget.deleteLater()
        self.camera_views.clear()
        self.camera_threads.clear()

        # Handle layout based on number of cameras
        for idx, cam_id in enumerate(cameras):
            thread = VideoThread(cam_id, self)  # Pass self as parent
            thread.frame_data.connect(self.update_frame)
            thread.attendance_updated.connect(self.on_attendance_updated)
            self.camera_threads[cam_id] = thread

            label = QLabel()
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)  # Use Preferred to limit growth
            if len(cameras) == 1:
                # Single camera: fill the entire video_widget
                self.video_layout.addWidget(label)
            else:
                # Two cameras: split vertically into two horizontal tabs
                self.video_layout.addWidget(label, stretch=1)  # Equal height

            self.camera_views[cam_id] = label
            self._adjust_label_size(label)  # Set initial maximum size
            thread.start()

        self.update_today_present_list()

    def stop_attendance(self):
        for thread in self.camera_threads.values():
            thread.stop()
        self.camera_threads.clear()

        for label in self.camera_views.values():
            self.video_layout.removeWidget(label)
            label.deleteLater()
        self.camera_views.clear()

        self.update_today_present_list()

    @Slot(str)
    def on_attendance_updated(self, name):
        if name not in self.present_today:
            self.present_today.add(name)
            self.update_today_present_list()

    def update_today_present_list(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.present_today.clear()

        for row in ws.iter_rows(min_row=2, values_only=True):
            name, date_str, _ = row
            if date_str == today and name not in self.present_today:
                self.present_today.add(name)

        self.ui.Attendance.setRowCount(len(self.present_today))
        for row, name in enumerate(sorted(self.present_today)):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.ui.Attendance.setItem(row, 0, item)

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

        email, ok = QInputDialog.getText(self, "Email Address", f"Enter email for {name}:")
        if not ok or not email.strip():
            return
        email = email.strip()

        self.save_email(name, email)

        folder_path = os.path.join("known_faces", name)
        os.makedirs(folder_path, exist_ok=True)

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
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if not file_path:
                return
            frame = cv2.imread(file_path)

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
        confirm = QMessageBox.question(self, "Confirm Deletion", f"Delete all data for {name}?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            shutil.rmtree(folder_path)
            self.delete_email(name)
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

    def get_absentees_today(self):
        all_people = set(os.listdir("known_faces"))
        present_today = set()
        today = datetime.now().strftime("%Y-%m-%d")
        for row in ws.iter_rows(min_row=2, values_only=True):
            name, date_str, _ = row
            if date_str == today:
                present_today.add(name)
        return list(all_people - present_today)

    def save_email(self, name, email):
        email_file = "emails.json"
        if os.path.exists(email_file):
            with open(email_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        data[name] = email
        with open(email_file, "w") as f:
            json.dump(data, f, indent=4)

    def delete_email(self, name):
        email_file = "emails.json"
        if os.path.exists(email_file):
            with open(email_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
            if name in data:
                del data[name]
                with open(email_file, "w") as f:
                    json.dump(data, f, indent=4)

    def send_absentee_emails(self):
        absentees = self.get_absentees_today()
        if not absentees:
            QMessageBox.information(self, "Info", "No absentees today!")
            return
        email_file = "emails.json"
        if os.path.exists(email_file):
            with open(email_file, "r") as f:
                try:
                    emails = json.load(f)
                except json.JSONDecodeError:
                    emails = {}
        else:
            emails = {}
        SMTP_SERVER = 'smtp.gmail.com'
        SMTP_PORT = 587
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
        except Exception as e:
            QMessageBox.warning(self, "SMTP Error", f"Failed to connect to SMTP server:\n{e}")
            return
        for name in absentees:
            recipient_email = emails.get(name)
            if not recipient_email:
                print(f"No email found for {name}, skipping.")
                continue
            msg = EmailMessage()
            msg['Subject'] = "Attendance Alert"
            msg['From'] = SMTP_USER
            msg['To'] = recipient_email
            msg.set_content(f"Dear {name},\n\nOur records show you were absent today ({datetime.now().strftime('%Y-%m-%d')}). Please contact your supervisor if this is an error.\n\nBest regards,\nAttendance System")
            try:
                server.send_message(msg)
                print(f"Sent absentee email to {name} at {recipient_email}")
            except Exception as e:
                print(f"Failed to send email to {name} ({recipient_email}): {e}")
        server.quit()
        QMessageBox.information(self, "Emails Sent", "Absentee emails have been sent.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
