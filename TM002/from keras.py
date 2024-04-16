import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt  # Qt 모듈 추가
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.upload_button = QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.uploadImage)

        self.classify_button = QPushButton('Classify Image')
        self.classify_button.clicked.connect(self.classifyImage)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.upload_button)
        vbox.addWidget(self.classify_button)

        self.setLayout(vbox)

        # Load the model
        self.model = load_model("./model/keras_Model.h5", compile=False)

        # Load the labels
        self.class_names = open("./model/labels.txt", "r").readlines()

    def uploadImage(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)", options=options)
        if filename:
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_path = filename

    def classifyImage(self):
        try:
            image = Image.open(self.image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            prediction = self.model.predict(data)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]
            QMessageBox.information(self, "Prediction Result", f"Class: {class_name[2:]}\nConfidence Score: {confidence_score}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())
