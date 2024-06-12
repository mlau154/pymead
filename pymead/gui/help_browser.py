import os

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QToolBar, QLineEdit, QWidget, QVBoxLayout, QToolButton, QDialog

from pymead import ICON_DIR


class HelpBrowserToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.back_icon = QIcon(os.path.join(ICON_DIR, "back.png"))
        self.forward_icon = QIcon(os.path.join(ICON_DIR, "forward.png"))
        self.reload_icon = QIcon(os.path.join(ICON_DIR, "reload.png"))
        self.home_icon = QIcon(os.path.join(ICON_DIR, "home.png"))
        self.stop_icon = QIcon(os.path.join(ICON_DIR, "stop.png"))
        self.back_button = QToolButton(parent=self)
        self.forward_button = QToolButton(parent=self)
        self.reload_button = QToolButton(parent=self)
        self.home_button = QToolButton(parent=self)
        self.url_bar = QLineEdit(parent=self)
        self.stop_button = QToolButton(parent=self)
        self.back_button.setIcon(self.back_icon)
        self.forward_button.setIcon(self.forward_icon)
        self.reload_button.setIcon(self.reload_icon)
        self.home_button.setIcon(self.home_icon)
        self.stop_button.setIcon(self.stop_icon)
        for w in [self.back_button, self.forward_button, self.reload_button, self.home_button, self.url_bar,
                  self.stop_button]:
            self.addWidget(w)


class HelpBrowser(QWebEngineView):
    """
    Help browser as inspired by https://www.geeksforgeeks.org/creating-a-simple-browser-using-pyqt5/
    """
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setUrl(QUrl("https://pymead.readthedocs.io/en/latest/gui.html"))


class HelpBrowserWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.toolbar = HelpBrowserToolBar(self)
        self.lay.addWidget(self.toolbar)
        self.help_browser = HelpBrowser(self)
        self.lay.addWidget(self.help_browser)
        self.help_browser.urlChanged.connect(self.update_urlbar)

        # Connections
        self.toolbar.back_button.clicked.connect(self.help_browser.back)
        self.toolbar.forward_button.clicked.connect(self.help_browser.forward)
        self.toolbar.reload_button.clicked.connect(self.help_browser.reload)
        self.toolbar.home_button.clicked.connect(self.navigate_home)
        self.toolbar.url_bar.returnPressed.connect(self.navigate_to_url)
        self.toolbar.stop_button.clicked.connect(self.help_browser.stop)

    # method called by the home action
    def navigate_home(self):

        # open the google
        self.help_browser.setUrl(QUrl("https://pymead.readthedocs.io/en/latest/gui.html"))

    # method called by the line edit when return key is pressed
    def navigate_to_url(self):

        # getting url and converting it to QUrl object
        q = QUrl(self.toolbar.url_bar.text())

        # if url is scheme is blank
        if q.scheme() == "":
            # set url scheme to html
            q.setScheme("http")

        # set the url to the browser
        self.help_browser.setUrl(q)

    # method for updating url
    # this method is called by the QWebEngineView object
    def update_urlbar(self, q):

        # setting text to the url bar
        self.toolbar.url_bar.setText(q.toString())

        # setting cursor position of the url bar
        self.toolbar.url_bar.setCursorPosition(0)


class HelpBrowserWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Help Browser")
        if self.parent() is not None:
            self.setFont(self.parent().font())
        self.setGeometry(100, 50, 800, 800)

        self.help_browser_widget = HelpBrowserWidget(self)
        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.lay.addWidget(self.help_browser_widget)
        self.show()
