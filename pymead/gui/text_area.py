from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor, QColor, QFontDatabase


class ConsoleTextArea(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        font = self.font()
        # print(QFontDatabase().families())
        font.setFamily("DejaVu Sans Mono")
        font.setPointSize(10)
        self.setFont(font)
        self.moveCursor(QTextCursor.End)
        self.setTextColor(QColor("#13294B"))
        self.setFixedHeight(200)