from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QTextEdit, QTextBrowser
from PyQt5.QtGui import QTextCursor, QColor, QFontDatabase, QDesktopServices


class ConsoleTextArea(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        font = self.font()
        # print(QFontDatabase().families())
        font.setFamily("DejaVu Sans Mono")
        # font.setFamily("Courier")
        font.setPointSize(10)
        self.setFont(font)
        self.moveCursor(QTextCursor.End)
        # self.setTextColor(QColor("#13294B"))
        self.setMinimumHeight(50)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)

        def handle_links(url):
            if not url.scheme():
                url = QUrl.fromLocalFile(url.toString())
            QDesktopServices.openUrl(url)

        self.anchorClicked.connect(handle_links)
