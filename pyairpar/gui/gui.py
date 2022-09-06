from mplcanvas import MplCanvas
from rename_popup import RenamePopup
from main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QWidget, QDoubleSpinBox, QLineEdit, QLabel, QMenu, QStatusBar
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtCore import QEvent, QObject

import pyqtgraph as pg
import numpy as np

from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.core.param import Param
from pyairpar.core.free_point import FreePoint
from pyairpar.gui.recalculate_airfoil_parameters import recalculate_airfoil_parameters
from pyairpar.gui.airfoil_graph import AirfoilGraph
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

        self.airfoils = [Airfoil()]
        self.airfoils[0].insert_free_point(FreePoint(Param(0.5), Param(0.1), previous_anchor_point='te_1', previous_free_point=None, name='upper_fp'))
        self.airfoils[0].update()
        self.airfoil_graphs = [AirfoilGraph(self.airfoils[0])]
        # print(f"Airfoil Curve 0 Handle = {self.airfoil_graph.airfoil.curve_list[0].pg_curve_handle}")
        self.w = self.airfoil_graphs[0].w
        self.v = self.airfoil_graphs[0].v
        #
        # self.airfoils.append(Airfoil())
        # self.airfoil_graphs.append(AirfoilGraph(self.airfoils[1], w=self.w, v=self.v, airfoil2=self.airfoils[1]))
        # self.airfoil_graph2 = None
        # print(f"Airfoil 2 Curve 0 Handle = {self.airfoil_graph2.airfoil.curve_list[0].pg_curve_handle}")
        # print(f"airfoil_graph_v = {self.airfoil_graph.v}")
        # print(f"airfoil_graph2_v = {self.airfoil_graph2.v}")
        # print(f"airfoil_graph_w = {self.airfoil_graph.w}")
        # print(f"airfoil_graph2_w = {self.airfoil_graph2.w}")

        self.main_layout = QHBoxLayout()
        self.create_design_tree()
        self.main_layout.addWidget(self.airfoil_graphs[0].w)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        # self.airfoil_graph.w.setFocus()
        self.setCentralWidget(self.main_widget)
        self.set_title_and_icon()
        self.main_icon_toolbar = MainIconToolbar(self)
        self.setStatusBar(QStatusBar(self))

    def set_title_and_icon(self):
        self.setWindowTitle("Airfoil Designer")
        image_path = os.path.join(os.path.dirname(os.getcwd()), 'icons', 'airfoil_slat.png')
        self.setWindowIcon(QIcon(image_path))

    def add_airfoil(self, airfoil: Airfoil = None):
        print(f"scene = {self.v.scene()}")
        if not airfoil:
            airfoil = Airfoil(
                base_airfoil_params=BaseAirfoilParams(dx=Param(-0.1), dy=Param(-0.2)))
        self.airfoil_graphs.append(AirfoilGraph(airfoil=airfoil, w=self.w, v=self.v))

    def create_design_tree(self):
        self.design_tree = QTreeWidget()
        self.design_tree.setGeometry(100, 100, 800, 500)
        self.design_tree.setColumnCount(1)
        self.design_tree.header().hide()
        items = [QTreeWidgetItem(self.design_tree)]
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
        for idx, curve in enumerate(self.airfoils[0].curve_list):
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


def main():
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    gui = GUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
