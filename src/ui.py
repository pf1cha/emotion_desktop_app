import logging
import re
from typing import Dict, List

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QProgressBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class EmotionDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор эмоций текста")
        self.resize(640, 590)

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

        self.model = None
        self.text_processor = None
        self.max_len = 64

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )

        logger.info("UI запущен в демо-режиме без загрузки модели и токенизатора.")
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
        self.predict_btn.clicked[bool].connect(self.on_predict_click)
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
        logger.info("Демо-режим: загрузка модели отключена.")
        return None, self.max_len

    def preprocess_text(self, text):
        """Заглушка препроцессинга: нормализует текст без внешних зависимостей."""
        normalized_text = (text or "").strip().lower()
        normalized_text = re.sub(r"\s+", " ", normalized_text)
        normalized_text = re.sub(r"[^a-z0-9\s'\-.,!?]", "", normalized_text)
        return normalized_text

    def predict_emotion(self, tensor_input):
        """Демонстрационный классификатор без нейросети."""
        text = self.preprocess_text(tensor_input)
        if not text:
            return [0.2] * len(self.classes)

        keyword_groups: Dict[int, List[str]] = {
            0: ["sad", "sadness", "lonely", "cry", "depressed", "sorry"],
            1: ["ok", "fine", "normal", "neutral", "average"],
            2: ["happy", "great", "good", "awesome", "love", "wonderful", "joy"],
            3: ["bad", "angry", "hate", "terrible", "awful", "mad"],
            4: ["think", "understand", "reason", "because", "analyze", "cognition"],
        }

        scores = [1.0] * len(self.classes)
        for class_id, keywords in keyword_groups.items():
            for keyword in keywords:
                if keyword in text:
                    scores[class_id] += 2.0

        total = sum(scores)
        if total <= 0:
            return [0.2] * len(self.classes)

        return [score / total for score in scores]

    def _format_result(self, probabilities):
        best_class_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
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

    def on_predict_click(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            return

        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("⏳ Обработка...")

        try:
            probabilities = self.predict_emotion(text)
            self._format_result(probabilities)
            logger.info("Демо-предсказание выполнено для текста длиной %s символов", len(text))

        except Exception as e:
            self.result_label.setText("❌ Ошибка")
            logger.exception("Ошибка во время демо-предсказания")

        finally:
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("🔍 Распознать эмоцию")
