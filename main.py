import sys
import torch
from PyQt6.QtWidgets import QApplication
from src.ui import EmotionDetectorApp


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    app.setStyleSheet("""
        QMainWindow {
            background-color: #ffffff;
        }
        QLabel {
            color: #2c3e50;
        }
    """)

    window = EmotionDetectorApp()
    window.show()
    sys.exit(app.exec())