import os

from PyQt5.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QIcon, QDesktopServices, QPalette, QLinearGradient, QGradient, QBrush
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton, QStyle

from pymead import ICON_DIR


class TitleBarButton(QToolButton):
    def __init__(self, parent, operation: str, theme: dict = None):
        super().__init__(parent)

        if operation not in ("minimize", "maximize", "normal", "close"):
            raise ValueError("Invalid operation")

        self.operation = operation
        self.theme = theme

    def hoverColor(self) -> str:
        if self.theme is not None:
            return self.theme[f"{self.operation}-hover-color"]
        else:
            return self.parent().parent().themes[self.parent().parent().current_theme][f"{self.operation}-hover-color"]

    def setColorDefault(self):
        theme_name = self.theme["theme-name"] if self.theme is not None else self.parent().parent().current_theme
        self.setIcon(QIcon(os.path.join(ICON_DIR, f"{self.operation}-{theme_name}-mode.svg")))
        self.setStyleSheet(f"""QToolButton {{ border: none }}""")

    def setColorHover(self):
        theme_name = self.theme["theme-name"] if self.theme is not None else self.parent().parent().current_theme
        if theme_name == "dark":
            self.setIcon(QIcon(os.path.join(ICON_DIR, f"{self.operation}-{theme_name}-mode.svg")))
        self.setStyleSheet(f"""QToolButton {{ background-color: {self.hoverColor()} }}""")

    def enterEvent(self, a0):
        self.setColorHover()
        super().enterEvent(a0)

    def leaveEvent(self, a0):
        self.setColorDefault()
        super().leaveEvent(a0)


class TitleBar(QWidget):
    clickPos = None

    sigMessage = pyqtSignal(str, str)

    def __init__(self, parent):
        super().__init__(parent)

        self.guiWindowMaximized = False

        theme = self.parent().themes[self.parent().current_theme]

        self.setAutoFillBackground(True)

        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(1, 1, 1, 1)

        self.title = QLabel("Custom Title Bar", self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFixedHeight(30)
        self.title.setAutoFillBackground(True)

        # style = self.style()
        # ref_size = self.fontMetrics().height()
        # ref_size += style.pixelMetric(style.PM_ButtonMargin) * 2
        # self.setMaximumHeight(ref_size + 2)

        # Add the pymead logo as a button that, when clicked, opens the pymead GitHub page in the machine's default
        # browser
        pymeadLogoButton = QToolButton(self)
        pymeadLogoButton.setIcon(QIcon(os.path.join(ICON_DIR, "pymead-logo.png")))
        pymeadLogoButton.setFixedSize(30, 30)
        pymeadLogoButton.setIconSize(QSize(30, 30))
        pymeadLogoButton.setToolTip("Open pymead GitHub repo")
        pymeadLogoButton.clicked.connect(self.openGitHubPage)

        gripFillerWidget = QWidget(self)
        gripFillerWidget.setFixedWidth(0)

        self.lay.addWidget(gripFillerWidget)
        self.lay.addWidget(pymeadLogoButton)
        self.lay.addWidget(self.title)

        # self.lay.addStretch()

        self.minimizeButton = None
        self.normalButton = None
        self.maximizeButton = None
        self.closeButton = None

        btn_size = QSize(30, 30)
        for target, picture, hover_color, tool_tip in zip(
                ('minimize', 'normal', 'maximize', 'close'),
                ("minimize-dark-mode.svg", "normal-dark-mode.svg", "maximize-dark-mode.svg", "close-dark-mode.svg"),
                (theme["minimize-hover-color"], theme["maximize-hover-color"],
                 theme["maximize-hover-color"], theme["close-hover-color"]),
                ("Minimize", "Normal", "Maximize", "Close")
        ):
            btn = TitleBarButton(self, operation=target)
            btn.setFocusPolicy(Qt.NoFocus)
            self.lay.addWidget(btn)
            btn.setFixedSize(btn_size)
            # btn.setIconSize(btn_size)

            btn.setIcon(QIcon(os.path.join(ICON_DIR, picture)))
            btn.setToolTip(tool_tip)

            btn.setStyleSheet('''QToolButton { border: none }''')

            signal = getattr(self, target + 'Clicked')
            btn.clicked.connect(signal)

            setattr(self, target + 'Button', btn)

        gripFillerWidget = QWidget(self)
        gripFillerWidget.setFixedWidth(0)

        self.lay.addWidget(gripFillerWidget)

        self.normalButton.hide()

        self.updateTitle(parent.windowTitle())
        parent.windowTitleChanged.connect(self.updateTitle)

    def updateTitle(self, title=None):
        if title is None:
            title = self.window().windowTitle()
        width = self.title.width()
        width -= self.style().pixelMetric(QStyle.PM_LayoutHorizontalSpacing) * 2
        self.title.setText(self.fontMetrics().elidedText(
            title, Qt.ElideRight, width))
        # self.title.setStyleSheet("QLabel { color: blue; }")

    def windowStateChanged(self, state):
        self.normalButton.setVisible(state == Qt.WindowMaximized)
        self.maximizeButton.setVisible(state != Qt.WindowMaximized)
        self.guiWindowMaximized = state == Qt.WindowMaximized

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.guiWindowMaximized:
            self.clickPos = event.windowPos().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.clickPos is not None and not self.guiWindowMaximized:
            self.window().move(event.globalPos() - self.clickPos)
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.clickPos = None
        super().mouseReleaseEvent(event)
        event.accept()

    def closeClicked(self):
        self.window().close()

    def maximizeClicked(self):
        self.window().showMaximized()
        self.maximizeButton.setColorDefault()

    def normalClicked(self):
        self.window().showNormal()
        self.normalButton.setColorDefault()

    def minimizeClicked(self):
        self.window().showMinimized()

    def openGitHubPage(self):
        url = QUrl("https://github.com/mlau154/pymead")
        if not QDesktopServices.openUrl(url):
            self.sigMessage.emit("Could not open pymead's GitHub page", "error")


class DialogTitleBar(QWidget):
    clickPos = None

    sigMessage = pyqtSignal(str, str)

    def __init__(self, parent, theme: dict):
        super().__init__(parent)

        self.guiWindowMaximized = False

        self.setAutoFillBackground(True)

        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(1, 1, 1, 1)

        self.title = QLabel("Custom Title Bar", self)
        self.title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title.setFixedHeight(30)
        self.title.setMinimumWidth(400)
        self.title.setAutoFillBackground(True)

        # style = self.style()
        # ref_size = self.fontMetrics().height()
        # ref_size += style.pixelMetric(style.PM_ButtonMargin) * 2
        # self.setMaximumHeight(ref_size + 2)

        # Add the pymead logo as a button that, when clicked, opens the pymead GitHub page in the machine's default
        # browser
        pymeadLogoButton = QToolButton(self)
        pymeadLogoButton.setIcon(QIcon(os.path.join(ICON_DIR, "pymead-logo.png")))
        pymeadLogoButton.setFixedSize(30, 30)
        pymeadLogoButton.setIconSize(QSize(30, 30))
        # pymeadLogoButton.setEnabled(False)

        gripFillerWidget = QWidget(self)
        gripFillerWidget.setFixedWidth(0)

        self.lay.addWidget(gripFillerWidget)
        self.lay.addWidget(pymeadLogoButton)
        self.lay.addWidget(self.title)

        # self.lay.addStretch()

        self.minimizeButton = None
        self.normalButton = None
        self.maximizeButton = None
        self.closeButton = None

        btn_size = QSize(30, 30)
        for target, picture, hover_color, tool_tip in zip(
                ('close',),
                (f"close-{theme['theme-name']}-mode.svg",),
                (theme["close-hover-color"],),
                ("Close",)
        ):
            btn = TitleBarButton(self, operation=target, theme=theme)
            btn.setFocusPolicy(Qt.NoFocus)
            self.lay.addWidget(btn)
            btn.setFixedSize(btn_size)
            # btn.setIconSize(btn_size)

            btn.setIcon(QIcon(os.path.join(ICON_DIR, picture)))
            btn.setToolTip(tool_tip)

            btn.setStyleSheet('''QToolButton { border: none }''')

            signal = getattr(self, target + 'Clicked')
            btn.clicked.connect(signal)

            setattr(self, target + 'Button', btn)

        gripFillerWidget = QWidget(self)
        gripFillerWidget.setFixedWidth(0)

        self.lay.addWidget(gripFillerWidget)

        self.updateTitle(parent.windowTitle())
        parent.windowTitleChanged.connect(self.updateTitle)

    def updateTitle(self, title=None):
        if title is None:
            title = self.window().windowTitle()
        width = self.title.width()
        width -= self.style().pixelMetric(QStyle.PM_LayoutHorizontalSpacing) * 2
        self.title.setText(self.fontMetrics().elidedText(
            title, Qt.ElideRight, width))
        # self.title.setStyleSheet("QLabel { color: blue; }")

    def windowStateChanged(self, state):
        self.normalButton.setVisible(state == Qt.WindowMaximized)
        self.maximizeButton.setVisible(state != Qt.WindowMaximized)
        self.guiWindowMaximized = state == Qt.WindowMaximized

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.guiWindowMaximized:
            self.clickPos = event.windowPos().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.clickPos is not None and not self.guiWindowMaximized:
            self.window().move(event.globalPos() - self.clickPos)
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.clickPos = None
        super().mouseReleaseEvent(event)
        event.accept()

    def closeClicked(self):
        self.window().close()

    def openGitHubPage(self):
        url = QUrl("https://github.com/mlau154/pymead")
        if not QDesktopServices.openUrl(url):
            self.sigMessage.emit("Could not open pymead's GitHub page", "error")
