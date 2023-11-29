from PyQt5.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel

from pymead.version import __version__


class PymeadProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setStyleSheet(
            '''
            QProgressBar {border: 2px solid; border-color: #8E9091;} 
            QProgressBar::chunk {background-color: #6495ED; width: 10px; margin: 0.5px;}
            '''
        )
        self.setValue(0)
        self.hide()


class PymeadInfoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText(f"pymead {__version__}")


class PermanentWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QHBoxLayout(self)
        self.progress_bar = PymeadProgressBar(self)
        self.info_label = PymeadInfoLabel(self)
        self.lay.addWidget(self.progress_bar)
        self.lay.addWidget(self.info_label)
        self.setLayout(self.lay)
