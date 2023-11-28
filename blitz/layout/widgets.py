from typing import Any

from PyQt5.QtWidgets import QTextEdit


class LoggingTextEdit(QTextEdit):

    def log(self, message: Any):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        if message != "\n":
            cursor.insertText(f"> {message}\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
