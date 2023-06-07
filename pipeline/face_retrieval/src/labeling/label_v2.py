import os
import json
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog


class ImageLabeler(QWidget):
    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        self.images = []
        self.current_image_index = 0

        self.labels = {}
        self.label_buttons = {}

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Labeler")

        # Load images from folder
        for file in os.listdir(self.folder):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                self.images.append(os.path.join(self.folder, file))

        # Load labels from JSON file if it exists
        json_file = os.path.join(self.folder, "labels.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                self.labels = json.load(f)

        # Create label buttons
        for label in sorted(set(self.labels.values())):
            button = QPushButton(label, self)
            button.clicked.connect(lambda _, l=label: self.label_image(l))
            self.label_buttons[label] = button

        # Create widgets
        self.image_label = QLabel(self)
        self.previous_button = QPushButton("Previous", self)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_image)
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_labels)

        # Layout widgets
        layout = self.layout()
        layout.addWidget(self.image_label, 0, 0, 3, 3)
        layout.addWidget(self.previous_button, 4, 0, 1, 1)
        layout.addWidget(self.next_button, 4, 1, 1, 1)
        layout.addWidget(self.save_button, 4, 2, 1, 1)

        for i, button in enumerate(self.label_buttons.values()):
            layout.addWidget(button, 5 + i // 3, i % 3, 1, 1)

        # Show first image
        self.show_image()

    def show_image(self):
        image_path = self.images[self.current_image_index]
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

        # Update label button states
        for label in set(self.labels.values()):
            self.label_buttons[label].setEnabled(True)

        if image_path in self.labels:
            self.label_buttons[self.labels[image_path]].setEnabled(False)

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image()

    def label_image(self, label):
        self.labels[self.images[self.current_image_index]] = label
        self.label_buttons[label].setEnabled(False)
        self.next_image()

    def save_labels(self):
        json_file = os.path.join(self.folder, "labels.json")
        with open(json_file, "w") as f:
            json.dump(self.labels, f, indent=4)

        self.save_button.setText("Saved!")


if __name__ == "__main__":
    app = QApplication([])
    folder = QFileDialog.getExistingDirectory(None, "Select a folder")
    if folder:
        labeler = ImageLabeler(folder)
        labeler.show()
        app.exec_()

