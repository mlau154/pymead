import os

from PyQt5.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton, QStyle

from pymead import ICON_DIR


class TitleBar(QWidget):
    clickPos = None

    sigMessage = pyqtSignal(str, str)

    def __init__(self, parent):
        super().__init__(parent)

        self.guiWindowMaximized = False

        self.setAutoFillBackground(True)

        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(1, 1, 1, 1)

        self.title = QLabel("Custom Title Bar", self)
        self.title.setAlignment(Qt.AlignCenter)

        style = self.style()
        ref_size = self.fontMetrics().height()
        ref_size += style.pixelMetric(style.PM_ButtonMargin) * 2
        self.setMaximumHeight(ref_size + 2)

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

        self.minButton = None
        self.normalButton = None
        self.maxButton = None
        self.closeButton = None

        btn_size = QSize(ref_size, ref_size)
        for target in ('min', 'normal', 'max', 'close'):
            btn = QToolButton(self)
            btn.setFocusPolicy(Qt.NoFocus)
            self.lay.addWidget(btn)
            btn.setFixedSize(btn_size)

            iconType = getattr(style,
                               'SP_TitleBar{}Button'.format(target.capitalize()))
            btn.setIcon(style.standardIcon(iconType))

            colorNormal = 'palette(mid)'
            colorHover = 'palette(light)'
            btn.setStyleSheet('''
                        QToolButton {{ border: none }}
                        QToolButton:hover {{
                            background-color: {}
                        }}
                    '''.format(colorNormal, colorHover))

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
        self.maxButton.setVisible(state != Qt.WindowMaximized)
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

    def maxClicked(self):
        self.window().showMaximized()

    def normalClicked(self):
        self.window().showNormal()

    def minClicked(self):
        self.window().showMinimized()

    def openGitHubPage(self):
        url = QUrl("https://github.com/mlau154/pymead")
        if not QDesktopServices.openUrl(url):
            self.sigMessage.emit("Could not open pymead's GitHub page", "error")

    # def resizeEvent(self, event):
    #     self.title.resize(self.minButton.x(), self.height())
    #     self.updateTitle()
