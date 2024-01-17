import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
import librosa
import numpy as np
import torch
from audio_signal_classification import SimpleCNN

# Sınıf etiketleri
result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Eğitilmiş modelin yolu ve yükleme
MODEL_PATH = 'cnn_model.pth'
model = torch.load(MODEL_PATH)
model.eval()

class AudioClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Classification")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label_result = QLabel("")

        self.button = QPushButton("Select WAV File")
        self.button.clicked.connect(self.open_file_dialog)

        layout.addWidget(self.button)
        layout.addWidget(self.label_result)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def classify_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=16000)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=60)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        input_data = torch.tensor(mfccs_scaled_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_data)
        predicted_class_index = torch.argmax(output).item()
        predicted_class = result_classes[predicted_class_index]
        return predicted_class

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select WAV File", "", "WAV files (*.wav)")
        if file_path:
            class_result = self.classify_audio(file_path)
            self.label_result.setText(f"Predicted Class: {class_result}")

def run_app():
    app = QApplication(sys.argv)
    window = AudioClassificationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
