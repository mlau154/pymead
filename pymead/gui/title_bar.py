import os
import sys

from PyQt6.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt6.QtGui import QIcon, QDesktopServices
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton, QStyle
from qframelesswindow.utils import startSystemMove

from pymead import ICON_DIR


class TitleBarButton(QToolButton):
    def __init__(self, parent, operation: str, theme: dict = None):
        super().__init__(parent)

        if operation not in ("minimize", "maximize", "normal", "close"):
            raise ValueError("Invalid operation")

        self.operation = operation
        self.theme = theme
        self.state = 0  # 0 = normal, 1 = hover, 2 = pressed

    def hoverColor(self) -> str:
        if self.theme is not None:
            return self.theme[f"{self.operation}-hover-color"]
        else:
            return self.parent().parent().themes[self.parent().parent().current_theme][f"{self.operation}-hover-color"]

    def setColorDefault(self):
        theme_name = self.theme["theme-name"] if self.theme is not None else self.parent().parent().current_theme
        self.setIcon(QIcon(os.path.join(ICON_DIR, f"{self.operation}-{theme_name}-mode.png")))
        self.setStyleSheet(f"""QToolButton {{ border: none }}""")

    def setColorHover(self):
        theme_name = self.theme["theme-name"] if self.theme is not None else self.parent().parent().current_theme
        if theme_name == "dark":
            self.setIcon(QIcon(os.path.join(ICON_DIR, f"{self.operation}-light-mode.png")))
        self.setStyleSheet(f"""QToolButton {{ background-color: {self.hoverColor()} }}""")

    def enterEvent(self, a0):
        self.setColorHover()
        self.state = 1
        super().enterEvent(a0)

    def leaveEvent(self, a0):
        self.setColorDefault()
        self.state = 0
        super().leaveEvent(a0)

    def mousePressEvent(self, a0):
        if a0.button() != Qt.MouseButton.LeftButton:
            return

        self.state = 2
        super().mousePressEvent(a0)

    def mouseReleaseEvent(self, a0):
        self.state = 0
        super().mouseReleaseEvent(a0)


class TitleBar(QWidget):
    clickPos = None

    sigMessage = pyqtSignal(str, str)

    def __init__(self, parent, theme: dict, window_title: str):
        super().__init__(parent)

        self.guiWindowMaximized = False

        self._isDoubleClickEnabled = True

        self.setAutoFillBackground(True)

        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(1, 1, 1, 1)

        self.title = QLabel("Custom Title Bar", self)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
                ("minimize-dark-mode.png", "normal-dark-mode.png", "maximize-dark-mode.png", "close-dark-mode.png"),
                (theme["minimize-hover-color"], theme["maximize-hover-color"],
                 theme["maximize-hover-color"], theme["close-hover-color"]),
                ("Minimize", "Normal", "Maximize", "Close")
        ):
            btn = TitleBarButton(self, operation=target)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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

        self.updateTitle(window_title)

    def updateTitle(self, title=None):
        if title is None:
            title = self.window().windowTitle()
        width = self.title.width()
        width -= self.style().pixelMetric(QStyle.PixelMetric.PM_LayoutHorizontalSpacing) * 2
        self.title.setText(self.fontMetrics().elidedText(
            title, Qt.TextElideMode.ElideRight, width))
        # self.title.setStyleSheet("QLabel { color: blue; }")

    def windowStateChanged(self, state):
        self.normalButton.setVisible(state == Qt.WindowState.WindowMaximized)
        self.maximizeButton.setVisible(state != Qt.WindowState.WindowMaximized)
        self.guiWindowMaximized = state == Qt.WindowState.WindowMaximized

    def mousePressEvent(self, event):
        if sys.platform == "win32" or not self.canDrag(event.pos()):
            return

        startSystemMove(self.window(), event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event):
        """ Toggles the maximization state of the window """
        if event.button() != Qt.MouseButton.LeftButton or not self._isDoubleClickEnabled:
            return

        self.__toggleMaxState()

    def mouseMoveEvent(self, event):
        if sys.platform != "win32" or not self.canDrag(event.pos()):
            return

        startSystemMove(self.window(), event.globalPosition().toPoint())

    def mouseReleaseEvent(self, event):
        self.clickPos = None
        super().mouseReleaseEvent(event)
        event.accept()

    def __toggleMaxState(self):
        """ Toggles the maximization state of the window and change icon """
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()

        if sys.platform == "win32":
            from qframelesswindow.utils.win32_utils import releaseMouseLeftButton
            releaseMouseLeftButton(self.window().winId())

    def _isDragRegion(self, pos):
        """ Check whether the position belongs to the area where dragging is allowed """
        width = 0
        for button in self.findChildren(TitleBarButton):
            if button.isVisible():
                width += button.width()

        return 0 < pos.x() < self.width() - width

    def _hasButtonPressed(self):
        """ whether any button is pressed """
        return any(btn.state == 2 for btn in self.findChildren(TitleBarButton))

    def canDrag(self, pos):
        """ whether the position is draggable """
        return self._isDragRegion(pos) and not self._hasButtonPressed()

    def setDoubleClickEnabled(self, isEnabled):
        """ whether to switch window maximization status when double-clicked

        Parameters
        ----------
        isEnabled: bool
            whether to enable double click
        """
        self._isDoubleClickEnabled = isEnabled

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
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
                (f"close-{theme['theme-name']}-mode.png",),
                (theme["close-hover-color"],),
                ("Close",)
        ):
            btn = TitleBarButton(self, operation=target, theme=theme)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        width -= self.style().pixelMetric(QStyle.PixelMetric.PM_LayoutHorizontalSpacing) * 2
        self.title.setText(self.fontMetrics().elidedText(
            title, Qt.TextElideMode.ElideRight, width))
        # self.title.setStyleSheet("QLabel { color: blue; }")

    def windowStateChanged(self, state):
        self.normalButton.setVisible(state == Qt.WindowState.WindowMaximized)
        self.maximizeButton.setVisible(state != Qt.WindowState.WindowMaximized)
        self.guiWindowMaximized = state == Qt.WindowState.WindowMaximized

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.guiWindowMaximized:
            self.clickPos = event.scenePosition().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.clickPos is not None and not self.guiWindowMaximized:
            self.window().move(event.globalPosition().toPoint() - self.clickPos)
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
