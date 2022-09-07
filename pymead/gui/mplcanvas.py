import matplotlib
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QWidget
from PyQt5 import QtCore
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, main_window: QMainWindow, width=5, height=4, dpi=100, xlabel="", ylabel="", title="",
                 xlim: tuple or None = None, ylim: tuple or None = None,
                 line_placeholders: typing.Tuple[dict, ...] or None = None,):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        if xlim is not None:
            self.axes.set_xlim(xlim)
        if ylim is not None:
            self.axes.set_ylim(ylim)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.suptitle(title)
        fig.set_tight_layout('tight')
        super(MplCanvas, self).__init__(fig)
        self.toolbar = NavigationToolbar(self, main_window)
        self.widget_layout = QVBoxLayout()
        self.widget_layout.addWidget(self.toolbar, alignment=QtCore.Qt.AlignCenter)
        self.widget_layout.addWidget(self, alignment=QtCore.Qt.AlignCenter)
        self.widget = QWidget()
        self.widget.setLayout(self.widget_layout)
