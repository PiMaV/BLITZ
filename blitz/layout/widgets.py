from typing import Any

from PyQt5.QtWidgets import (QDialog, QLabel, QMainWindow, QTextEdit,
                             QVBoxLayout)


class LoadingDialog:

    def __init__(self, main_window: QMainWindow, message: str) -> None:
        self.main_window = main_window
        self.dialog = QDialog()
        self.dialog.setWindowTitle(message)
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        self.dialog.setLayout(layout)
        self.dialog.setModal(True)


class LoggingTextEdit(QTextEdit):

    def log(self, message: Any):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        if message != "\n":
            cursor.insertText(f"> {message}\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
