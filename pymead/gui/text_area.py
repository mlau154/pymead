from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QTextEdit, QTextBrowser
from PyQt6.QtGui import QTextCursor, QDesktopServices


class ConsoleTextArea(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        # font = self.font()
        # print(QFontDatabase().families())
        # font.setFamily("DejaVu Sans Mono")
        # # font.setFamily("Courier")
        # font.setPointSize(6)
        # self
        # self.setFont(font)
        # self.setStyleSheet("""font: 10pt DejaVu Sans Mono;""")
        # prepend_html = f"<head><style>p {{font-family: DejaVu Sans Mono; font-size: 10pt}}</style>" \
        #                f"</head><body><p>&#8203;</p></body>"
        self.moveCursor(QTextCursor.MoveOperation.End)
        # self.setTextColor(QColor("#13294B"))
        # self.setFontPointSize(5)
        self.setMinimumHeight(50)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(True)

        self.document().setDefaultStyleSheet("""p, a {font-family: 'DejaVu Sans Mono'; font-size: 10pt}
                                                        a:link {color: orange;} 
                                                        a:active {color: green;} 
                                                        a:visited {color: hotpink;}
                                                        a:hover: {color: blue;}
                                                     """)

        def handle_links(url):
            if not url.scheme():
                url = QUrl.fromLocalFile(url.toString())
            QDesktopServices.openUrl(url)

        self.anchorClicked.connect(handle_links)
