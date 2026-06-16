
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QListWidget
)


class JarvisUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jarvis")
        self.resize(1200, 800)

        self.setStyleSheet("""
        QWidget {
            background: #121212;
            color: white;
            font-family: Segoe UI;
        }

        QFrame#Card {
            background: #1d1d1d;
            border-radius: 18px;
        }

        QListWidget {
            background: #181818;
            border: none;
            border-radius: 16px;
            padding: 10px;
        }

        QListWidget::item {
            padding: 12px;
            border-radius: 8px;
        }

        QListWidget::item:selected {
            background: #2b2b2b;
        }

        QTextEdit {
            background: #181818;
            border: none;
            border-radius: 16px;
            padding: 12px;
            font-size: 14px;
        }

        QPushButton {
            background: #2f6df6;
            border: none;
            border-radius: 12px;
            padding: 10px;
            font-size: 14px;
        }

        QPushButton:hover {
            background: #4a82ff;
        }
        """)

        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        main.setContentsMargins(15, 15, 15, 15)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Card")
        sidebar.setFixedWidth(250)

        side_layout = QVBoxLayout(sidebar)

        title = QLabel("JARVIS")
        title.setStyleSheet("font-size:24px;font-weight:bold;")
        side_layout.addWidget(title)

        menu = QListWidget()
        menu.addItems(["New Chat", "History", "Settings"])
        side_layout.addWidget(menu)

        main.addWidget(sidebar)

        # Main area
        content = QFrame()
        content.setObjectName("Card")

        content_layout = QVBoxLayout(content)

        header = QHBoxLayout()

        app_title = QLabel("Jarvis Voice Assistant")
        app_title.setStyleSheet("font-size:22px;font-weight:bold;")

        self.status = QLabel("● Waiting")
        self.status.setStyleSheet("""
        background:#1f3b24;
        color:#6ee787;
        padding:8px 16px;
        border-radius:12px;
        """)

        header.addWidget(app_title)
        header.addStretch()
        header.addWidget(self.status)

        content_layout.addLayout(header)

        # Mic orb
        orb_container = QVBoxLayout()

        self.orb = QLabel("🎤")
        self.orb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orb.setFixedSize(180, 180)
        self.orb.setStyleSheet("""
        font-size:72px;
        background:#2f6df6;
        border-radius:90px;
        """)

        orb_container.addSpacing(20)
        orb_container.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignCenter)

        self.state_label = QLabel("Waiting for wake word...")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size:18px;")

        orb_container.addWidget(self.state_label)

        content_layout.addLayout(orb_container)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)

        self.chat.append("Jarvis: Ready.")
        self.chat.append("")
        self.chat.append("You: Hello Jarvis")
        self.chat.append("")
        self.chat.append("Jarvis: How can I help you today?")

        content_layout.addWidget(self.chat)

        buttons = QHBoxLayout()

        start_btn = QPushButton("Start Listening")
        stop_btn = QPushButton("Stop")

        buttons.addWidget(start_btn)
        buttons.addWidget(stop_btn)

        content_layout.addLayout(buttons)

        main.addWidget(content)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JarvisUI()
    window.show()
    sys.exit(app.exec())
