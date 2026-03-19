import os
import logging
import numpy as np
import pickle
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from src.model_inference import BiLSTMClassifier
from src.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class EmotionDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор тональности")
        self.resize(600, 550)

        self.emotion_colors = {
            0: "#5dade2",  # Sadness
            1: "#95a5a6",  # Neutral
            2: "#2ecc71",  # Positive
            3: "#e74c3c",  # Negative
            4: "#9b59b6",  # Cognition
        }

        self.classes = {
            0: "Sadness",
            1: "Neutral",
            2: "Positive",
            3: "Negative",
            4: "Cognition",
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.text_processor = None
        self.max_len = 64

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )

        try:
            self.model, self.max_len = self.load_model_and_params()
            self.text_processor = TextProcessor(max_len=self.max_len)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Модель загружена на устройстве: %s", self.device)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить модель:\n{str(e)}")
            logger.exception("Ошибка при загрузке модели")
            self.model = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("🎭 Анализатор Эмоций Текста")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        self.input_label = QLabel("Введите сообщение на английском:")
        self.input_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        main_layout.addWidget(self.input_label)

        self.text_input = QTextEdit()
        self.text_input.setFont(QFont("Arial", 11))
        self.text_input.setMaximumHeight(100)
        main_layout.addWidget(self.text_input)

        self.predict_btn = QPushButton("🔍 Распознать эмоцию")
        self.predict_btn.setMinimumHeight(45)
        self.predict_btn.clicked.connect(self.on_predict_click)
        main_layout.addWidget(self.predict_btn)

        self.result_label = QLabel("Эмоция: ...")
        self.result_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)

        probs_label = QLabel("Уверенность модели по классам (независимая):")
        main_layout.addWidget(probs_label)

        self.progress_bars = {}
        probs_layout = QVBoxLayout()
        for class_id, class_name in self.classes.items():
            row_layout = QHBoxLayout()
            lbl = QLabel(class_name)
            lbl.setFixedWidth(120)

            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setFixedHeight(25)
            pb.setStyleSheet(f"QProgressBar::chunk {{background-color: {self.emotion_colors[class_id]};}}")

            row_layout.addWidget(lbl)
            row_layout.addWidget(pb)
            probs_layout.addLayout(row_layout)
            self.progress_bars[class_id] = pb
        main_layout.addLayout(probs_layout)

    def load_model_and_params(self):
        logger.info("Загрузка модели и параметров...")
        params_path = "src/models/model_params.pkl"
        weights_path = "src/models/best_emotion_model.pth"

        with open(params_path, "rb") as f:
            params = pickle.load(f)

        hidden_dim = params.get("hidden_dim", 128)
        num_layers = params.get("num_layers", 2)
        num_classes = params.get("num_classes", 6)
        max_len = params.get("max_len", 64)
        embedding_matrix = params.get("embedding_matrix")

        if embedding_matrix is None:
            raise ValueError("Матрица эмбеддингов не найдена в pickle файле!")

        pretrained_embeddings = torch.FloatTensor(embedding_matrix)

        model = BiLSTMClassifier(
            pretrained_embeddings=pretrained_embeddings,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        logger.info("Модель успешно загружена.")
        return model, max_len

    def preprocess_text(self, text):
        if self.model is None or self.text_processor is None:
            return None
        return self.text_processor.get_input_ids(text)

    def predict_emotion(self, tensor_input):
        if self.model is None or tensor_input is None:
            return np.zeros(len(self.classes), dtype=float)

        with torch.no_grad():
            tensor_input = tensor_input.to(self.device)
            logits = self.model(tensor_input)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        shown_classes_count = len(self.classes)
        if probs.shape[0] >= shown_classes_count:
            probs = probs[:shown_classes_count]
        else:
            probs = np.pad(probs, (0, shown_classes_count - probs.shape[0]), mode="constant")

        return probs

    def on_predict_click(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            return

        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("⏳ Обработка...")

        try:
            input_ids = self.preprocess_text(text)
            probabilities = self.predict_emotion(input_ids)

            best_class_idx = int(np.argmax(probabilities))
            best_emotion = self.classes.get(best_class_idx, "Неизвестно")
            confidence = probabilities[best_class_idx] * 100

            if confidence < 30:
                self.result_label.setText("🤔 Эмоция не выражена ярко")
                self.result_label.setStyleSheet("color: #7f8c8d;")
            else:
                self.result_label.setText(f"🎯 Главная эмоция: {best_emotion} ({confidence:.1f}%)")
                self.result_label.setStyleSheet("color: #2c3e50;")

            for class_id, pb in self.progress_bars.items():
                percentage = int(probabilities[class_id] * 100)
                pb.setValue(percentage)

        except Exception as e:
            self.result_label.setText("❌ Ошибка")
            logger.exception("Ошибка во время предсказания")

        finally:
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("🔍 Распознать эмоцию")
