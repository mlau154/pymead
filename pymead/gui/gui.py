from mplcanvas import MplCanvas
from rename_popup import RenamePopup
from main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, \
    QWidget, QDoubleSpinBox, QLineEdit, QLabel, QMenu, QStatusBar, QAction, QToolButton
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtCore import QEvent, QObject
from functools import partial

import pyqtgraph as pg
import numpy as np

import pickle
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead.core.free_point import FreePoint
from pymead.gui.recalculate_airfoil_parameters import recalculate_airfoil_parameters
from pymead.gui.airfoil_graph import AirfoilGraph
from pymead.gui.parameter_tree import MEAParamTree
from pymead.core.mea import MEA
from pymead.gui.parameter_tree import HeaderParameter
from draggable_line import DraggableLine

import sys
import os
# import matplotlib
# matplotlib.use('Qt5Agg')


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.design_tree = None

        self.mea = MEA([Airfoil(), Airfoil(), Airfoil()], airfoil_graphs_active=True)
        # self.mea.airfoils['A0'].insert_free_point(FreePoint(Param(0.5), Param(0.1), previous_anchor_point='te_1'))
        # self.mea.airfoils['A0'].update()
        # self.airfoil_graphs = [AirfoilGraph(self.mea.airfoils['A0'])]
        self.w = self.mea.airfoils['A0'].airfoil_graph.w
        self.v = self.mea.airfoils['A0'].airfoil_graph.v
        # self.airfoil_graphs.append(AirfoilGraph(self.mea.airfoils['A1'], w=self.w, v=self.v))
        self.main_layout = QHBoxLayout()
        self.setStatusBar(QStatusBar(self))
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar())
        # print(f"param_tree_instance = {self.param_tree_instance}")
        self.design_tree_widget = self.param_tree_instance.t
        self.main_layout.addWidget(self.design_tree_widget)
        self.main_layout.addWidget(self.w)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        # print(f"children of gui = {self.main_widget.children()}")
        # self.airfoil_graph.w.setFocus()
        self.setCentralWidget(self.main_widget)
        self.set_title_and_icon()
        self.create_menu_bar()
        self.main_icon_toolbar = MainIconToolbar(self)

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(os.path.dirname(os.getcwd()), 'icons', 'airfoil_slat.png')
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        # print(self.menu_bar)
        self.menu_names = {"&File": ["&Open", "&Save"]}
        # def recursively_add_menus(menu: dict, menu_bar: QMenu):
        #     for key, val in menu.items():
        #         if isinstance(val, dict):
        #             menu_bar.addMenu(QMenu(key, self))
        #             recursively_add_menus(val, menu_bar.children()[0])
        #         else:
        #
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.open_action = QAction("Open", self)
        self.file_menu.addAction(self.open_action)
        self.open_action.triggered.connect(self.load_mea)

        self.save_action = QAction("Save", self)
        self.file_menu.addAction(self.save_action)
        self.save_action.triggered.connect(self.save_mea)

    def save_mea(self):
        with open(os.path.join(os.getcwd(), 'test_mea.mead'), "wb") as f:
            pickle.dump(self.mea, f)
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, w=self.w, v=self.v)
        for a_name, a in self.mea.airfoils.items():
            # print(f"airfoil_name = {a_name}")
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed, a_name))

    def load_mea(self):
        with open(os.path.join(os.getcwd(), 'test_mea.mead'), "rb") as f:
            self.mea = pickle.load(f)
        self.v.clear()
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, w=self.w, v=self.v)
        # self.param_tree_instance.mea = self.mea
        # self.param_tree_instance.params[-1].clearChildren()
        # self.param_tree_instance.params[-1].airfoil_headers = []
        # print(f"vars = {vars(self.param_tree_instance.p)}")
        # self.param_tree_instance.custom_header = self.param_tree_instance.params[-1].addChild(
        #     HeaderParameter(name='Custom', type='bool', value=True))
        # print(self.param_tree_instance.params[-1].children())
        # for idx, a in enumerate(self.mea.airfoils.values()):
        #     self.param_tree_instance.params[-1].add_airfoil(a, idx)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar())
        self.design_tree_widget = self.param_tree_instance.t
        # print(f"children of gui = {self.main_widget.children()}")
        # print(f"design_tree gui = {self.design_tree_widget}")
        # print(f"design_tree gui 2 = {self.param_tree_instance}")
        # for a_name, a in self.mea.airfoils.items():
        #     # print(f"airfoil_name = {a_name}")
        #     a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed, a_name))
        # print(f"mea gui = {self.mea}")
        # print(f"mea param_tree = {self.param_tree_instance.mea}")
        # self.main_layout.addWidget(self.design_tree_widget)
        # self.main_layout.addWidget(self.w)
        # self.main_widget = QWidget()
        # self.main_widget.setLayout(self.main_layout)
        # print(f"children of gui = {self.main_widget.children()}")
        # # self.airfoil_graph.w.setFocus()
        # self.setCentralWidget(self.main_widget)
        self.main_layout.replaceWidget(self.main_layout.itemAt(0).widget(), self.design_tree_widget)
        # print(f"v= {self.v}")
        self.v.autoRange()

    def add_airfoil(self, airfoil: Airfoil = None):
        if not airfoil:
            airfoil = Airfoil(
                base_airfoil_params=BaseAirfoilParams(dx=Param(-0.1), dy=Param(-0.2)))
        self.airfoil_graphs.append(AirfoilGraph(airfoil=airfoil, w=self.w, v=self.v))

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
    app.setStyle('Fusion')
    gui = GUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
