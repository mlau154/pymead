from mplcanvas import MplCanvas
from rename_popup import RenamePopup
from main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QWidget, QDoubleSpinBox, QLineEdit, QLabel, QMenu, QStatusBar
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtCore import QEvent, QObject

import pyqtgraph as pg
import numpy as np

from pyairpar.core.airfoil import Airfoil
from pyairpar.gui.recalculate_airfoil_parameters import recalculate_airfoil_parameters
from draggable_line import DraggableLine

import sys
import os
import matplotlib
matplotlib.use('Qt5Agg')


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.drs = []  # Extremely important that this be left as an instance variable since Qt will garbage collect
        # the draggable line otherwise
        self.design_tree = None
        self.airfoil = Airfoil()

        self.mplcanvas1 = MplCanvas(self, width=12, height=6)
        self.main_layout = QHBoxLayout()
        self.create_design_tree()
        self.main_layout.addWidget(self.mplcanvas1.widget)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.mplcanvas1.widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.set_title_and_icon()
        self.plot_airfoil_on_canvas(self.mplcanvas1)
        self.main_icon_toolbar = MainIconToolbar(self)
        self.setStatusBar(QStatusBar(self))

        # PyQtGraph test
        win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
        win.resize(1000, 600)
        win.setWindowTitle('pyqtgraph example: Plotting')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
        win.setBackground('w')

        # pw = pg.PlotWidget()

        # plot_widget = pg.PlotWidget()

        b = win.addPlot(x=np.array([0, 1]), y=np.array([0, 1]), symbol='x')

        # b = win.addPlot(title="Basic array plotting", x=np.array([0, 1]), y=np.array([0, 1]), symbol='x')
        # b.set_movable(True)

        self.main_layout.addWidget(win)

    def set_title_and_icon(self):
        self.setWindowTitle("Airfoil Designer")
        image_path = os.path.join(os.path.dirname(os.getcwd()), 'icons', 'airfoil_slat.png')
        self.setWindowIcon(QIcon(image_path))

    def create_design_tree(self):
        self.design_tree = QTreeWidget()
        self.design_tree.setGeometry(100, 100, 800, 500)
        self.design_tree.setColumnCount(1)
        self.design_tree.header().hide()
        items = []
        items.append(QTreeWidgetItem(self.design_tree))
        items[0].setText(0, "Airfoils")
        child_item_airfoil = QTreeWidgetItem(items[0])
        child_item_airfoil.setText(0, "Airfoil 0")
        items[0].insertChild(0, child_item_airfoil)
        child_item_curve = QTreeWidgetItem(items[0].child(0))
        child_item_curve.setText(0, "Curves")
        items[0].child(0).insertChild(0, child_item_curve)
        # items = [f"item {i}" for i in range(10)]
        self.design_tree.insertTopLevelItems(0, items)
        # self.design_tree.setItemWidget(items[0], 0, QLineEdit('Airfoil 1'))
        # print(items)
        for idx, curve in enumerate(self.airfoil.curve_list):
            child_item = QTreeWidgetItem(items[0].child(0).child(0))
            items[0].child(0).child(0).insertChild(idx, child_item)

            child_item.setText(0, f"Curve {idx}")
            # line_edit.contextMenuEvent()
            # self.design_tree.setItemWidget(items[0].child(0).child(0).child(idx), 0, line_edit)
            # line_edit.mouseDoubleClickEvent(QMouseEvent())
        # print(items[2].child(0).text(0))
        # print('WEEEEEEEEEEEEEE')
        # self.design_tree.setItemWidget(items[2], 0, QDoubleSpinBox())
        self.design_tree.installEventFilter(self)
        # self.design_tree.resize(700, 500)
        self.main_layout.addWidget(self.design_tree)
        # self.setLayout(self.layout1)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ContextMenu and source is self.design_tree:
            menu = QMenu()
            menu.addAction('Rename')

            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                if item.text(0) not in ['Airfoils', 'Curves']:
                    rename_popup = RenamePopup(item.text(0), item)
                    rename_popup.exec()
            return True

        return super().eventFilter(source, event)

    def plot_airfoil_on_canvas(self, canvas: MplCanvas):
        airfoil = Airfoil()
        lines1 = airfoil.plot_airfoil(canvas.axes, color='cornflowerblue', lw=2, label='airfoil')
        curve_list2 = airfoil.plot_curvature_comb_normals(canvas.axes, 0.04, color='mediumaquamarine', lw=0.8)
        curve_list3 = airfoil.plot_curvature_comb_curve(canvas.axes, 0.04, color='indianred', lw=0.8)
        print(f'Normals = {airfoil.plt_normals}')
        curve_list4 = airfoil.plot_control_point_skeleton(canvas.axes, color='grey', ls='--', marker='*', lw=1.2)
        # for curve in curve_list1:
        #     curve.set_picker(7)
        #     # curve.pick()
        for curve in curve_list4:
            dr = DraggableLine(curve, 7, button_release_callback=recalculate_airfoil_parameters, airfoil=airfoil,
                               canvas=canvas, lines=lines1)
            dr.connect()
            self.drs.append(dr)
        canvas.axes.margins(x=0.1, y=1.5, tight=True)
        canvas.axes.set_aspect('equal')
        canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = GUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
