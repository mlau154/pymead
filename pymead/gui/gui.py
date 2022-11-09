from pymead.gui.rename_popup import RenamePopup
from pymead.gui.main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, \
    QWidget, QMenu, QStatusBar, QAction
from PyQt5.QtGui import QIcon, QFont, QFontDatabase
from PyQt5.QtCore import QEvent, QObject, Qt
from functools import partial

import pyqtgraph as pg
import numpy as np

import dill
from pymead.core.airfoil import Airfoil
from pymead import DATA_DIR, RESOURCE_DIR
from pymead.gui.input_dialog import SingleAirfoilViscousDialog, LoadDialog, SaveAsDialog
from pymead.gui.analysis_graph import AnalysisGraph
from pymead.gui.parameter_tree import MEAParamTree
from pymead.utils.airfoil_matching import match_airfoil
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.dockable_tab_widget import DockableTabWidget
from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data

import sys
import os


class GUI(QMainWindow):
    def __init__(self, path=None):
        # super().__init__(flags=Qt.FramelessWindowHint)
        super().__init__()
        self.path = path
        single_element_inviscid(np.array([[1, 0], [0, 0], [1, 0]]), 0.0)
        for font_name in ["DejaVuSans", "DejaVuSansMono", "DejaVuSerif"]:
            QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "dejavu-fonts-ttf-2.37", "ttf",
                                                          f"{font_name}.ttf"))
        # QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "cascadia-code", "Cascadia.ttf"))
        # print(QFontDatabase().families())

        self.design_tree = None
        self.dialog = None
        self.analysis_graph = None
        self.te_thickness_edit_mode = False
        self.dark_mode = False
        self.n_analyses = 0
        self.n_converged_analyses = 0
        self.pens = [('#d4251c', Qt.SolidLine), ('darkorange', Qt.SolidLine), ('gold', Qt.SolidLine),
                     ('limegreen', Qt.SolidLine), ('cyan', Qt.SolidLine), ('mediumpurple', Qt.SolidLine),
                     ('deeppink', Qt.SolidLine), ('#d4251c', Qt.DashLine), ('darkorange', Qt.DashLine),
                     ('gold', Qt.DashLine),
                     ('limegreen', Qt.DashLine), ('cyan', Qt.DashLine), ('mediumpurple', Qt.DashLine),
                     ('deeppink', Qt.DashLine)]
        # self.setFont(QFont("DejaVu Serif"))
        self.setFont(QFont("DejaVu Sans"))

        self.mea = MEA(None, [Airfoil()], airfoil_graphs_active=True)
        # self.mea.airfoils['A0'].insert_free_point(FreePoint(Param(0.5), Param(0.1), previous_anchor_point='te_1'))
        # self.mea.airfoils['A0'].update()
        # self.airfoil_graphs = [AirfoilGraph(self.mea.airfoils['A0'])]
        self.w = self.mea.airfoils['A0'].airfoil_graph.w
        self.v = self.mea.airfoils['A0'].airfoil_graph.v
        # internal_geometry_xy = np.loadtxt(os.path.join(DATA_DIR, 'sec_3.txt'))
        # # print(f"geometry = {internal_geometry_xy}")
        # scale_factor = 0.728967
        # x_start = 0.051039494
        # self.internal_geometry = self.v.plot(internal_geometry_xy[:, 0] * scale_factor + x_start,
        #                                      internal_geometry_xy[:, 1] * scale_factor,
        #                                      pen=pg.mkPen(color='orange', width=1))
        # self.airfoil_graphs.append(AirfoilGraph(self.mea.airfoils['A1'], w=self.w, v=self.v))
        self.main_layout = QHBoxLayout()
        self.setStatusBar(QStatusBar(self))
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        self.mea.airfoils['A0'].airfoil_graph.param_tree = self.param_tree_instance
        self.mea.airfoils['A0'].airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')
        # print(f"param_tree_instance = {self.param_tree_instance}")
        self.design_tree_widget = self.param_tree_instance.t
        # self.design_tree_widget.setAlternatingRowColors(False)
        # self.design_tree_widget.setStyleSheet("selection-background-color: #36bacfaa; selection-color: black;")
        # self.design_tree_widget.setStyleSheet('''QTreeWidget {color: black; alternate-background-color: red;
        #         selection-background-color: #36bacfaa;}
        #         QTreeView::item:hover {background: #36bacfaa;} QTreeView::item {border: 0px solid gray; color: black}''')
        self.setStyleSheet("color: black; font-family: DejaVu; font-size: 12px;")
        self.text_area = ConsoleTextArea()
        self.right_widget_layout = QVBoxLayout()
        # self.tab_widget = QTabWidget()
        # self.tab_widget.addTab(self.w, "Geometry")
        self.dockable_tab_window = DockableTabWidget(self)
        self.dockable_tab_window.add_new_tab_widget(self.w, "Geometry")

        self.right_widget_layout.addWidget(self.dockable_tab_window)
        self.right_widget_layout.addWidget(self.text_area)
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_widget_layout)
        self.main_layout.addWidget(self.design_tree_widget, 1)
        self.main_layout.addWidget(self.right_widget, 3)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        # print(f"children of gui = {self.main_widget.children()}")
        # self.airfoil_graph.w.setFocus()
        self.setCentralWidget(self.main_widget)

        self.set_title_and_icon()
        self.create_menu_bar()
        self.main_icon_toolbar = MainIconToolbar(self)
        if self.path is not None:
            self.load_mea_no_dialog(self.path)

    def set_dark_mode(self):
        self.setStyleSheet("background-color: #3e3f40; color: #dce1e6; font-family: DejaVu; font-size: 12px;")
        self.w.setBackground('#2a2a2b')

    def set_light_mode(self):
        self.setStyleSheet("font-family: DejaVu; font-size: 12px;")
        self.w.setBackground('w')

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

        # File Menu set-up
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.open_action = QAction("Open", self)
        self.file_menu.addAction(self.open_action)
        self.open_action.triggered.connect(self.load_mea)

        self.save_as_action = QAction("Save As", self)
        self.file_menu.addAction(self.save_as_action)
        self.save_as_action.triggered.connect(self.save_as_mea)

        self.save_action = QAction("Save", self)
        self.file_menu.addAction(self.save_action)
        self.save_action.triggered.connect(self.save_mea)

        self.settings_action = QAction("Settings", self)
        self.file_menu.addAction(self.settings_action)
        # self.settings_action.triggered.connect()

        # Analysis Menu set-up
        self.analysis_menu = QMenu("&Analysis", self)
        self.menu_bar.addMenu(self.analysis_menu)

        self.single_menu = QMenu("Single Airfoil", self)
        self.analysis_menu.addMenu(self.single_menu)
        self.multi_menu = QMenu("Multi-Element Airfoil", self)
        self.analysis_menu.addMenu(self.multi_menu)

        self.single_inviscid_action = QAction("Invisid", self)
        self.single_menu.addAction(self.single_inviscid_action)
        self.single_inviscid_action.triggered.connect(self.single_airfoil_inviscid_analysis)

        self.single_viscous_action = QAction("Viscous", self)
        self.single_menu.addAction(self.single_viscous_action)
        self.single_viscous_action.triggered.connect(self.single_airfoil_viscous_analysis)

        self.tools_menu = QMenu("&Tools", self)
        self.menu_bar.addMenu(self.tools_menu)

        self.match_airfoil_action = QAction("Match Airfoil", self)
        self.tools_menu.addAction(self.match_airfoil_action)
        self.match_airfoil_action.triggered.connect(self.match_airfoil)

    def save_as_mea(self):
        dialog = SaveAsDialog(self)
        if dialog.exec_():
            self.mea.file_name = dialog.selectedFiles()[0]
        else:
            pass
        self.save_mea()

    def save_mea(self):
        with open(os.path.join(os.getcwd(), self.mea.file_name), "wb") as f:
            dill.dump(self.mea, f)
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, self.param_tree_instance, w=self.w, v=self.v)
        for a_name, a in self.mea.airfoils.items():
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed, a_name))

    def load_mea(self):
        dialog = LoadDialog(self)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
        else:
            file_name = None
        if file_name is not None:
            self.load_mea_no_dialog(file_name)

    def load_mea_no_dialog(self, file_name):
        with open(os.path.join(os.getcwd(), file_name), "rb") as f:
            self.mea = dill.load(f)
        self.v.clear()
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, None, w=self.w, v=self.v)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        for a in self.mea.airfoils.values():
            a.airfoil_graph.param_tree = self.param_tree_instance
            a.airfoil_graph.airfoil_parameters = a.airfoil_graph.param_tree.p.param('Airfoil Parameters')
        self.design_tree_widget = self.param_tree_instance.t
        self.main_layout.replaceWidget(self.main_layout.itemAt(0).widget(), self.design_tree_widget)
        self.v.autoRange()

    def single_airfoil_inviscid_analysis(self):
        pass

    def single_airfoil_viscous_analysis(self):
        self.dialog = SingleAirfoilViscousDialog(items=[("Re", "double", 1e5), ("Iterations", "int", 150),
                                                        ("Timeout (seconds)", "double", 15),
                                                        ("Angle of Attack (degrees)", "double", 0.0),
                                                        ("Airfoil", "combo"), ("Name", "string", "default_airfoil"),
                                                        ("xtr upper", "double", 1.0), ("xtr lower", "double", 1.0),
                                                        ("Ncrit", "double", 9.0),
                                                        ("Use body-fixed CSYS", "checkbox", False)],
                                                 a_list=[k for k in self.mea.airfoils.keys()])
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None

        if inputs is not None:
            xfoil_settings = {'Re': inputs[0], 'timeout': inputs[2], 'iter': inputs[1], 'xtr': [inputs[6], inputs[7]],
                              'N': inputs[8]}
            aero_data, _ = calculate_aero_data(DATA_DIR, inputs[5], inputs[3], self.mea.airfoils[inputs[4]], 'xfoil',
                                               xfoil_settings, body_fixed_csys=inputs[9])
            if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
                self.text_area.insertPlainText(f"[{self.n_analyses:2.0f}] Converged = {aero_data['converged']} | Errored out = "
                                               f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}\n")
            else:
                self.text_area.insertPlainText(f"[{self.n_analyses:2.0f}] {inputs[4]} (\u03b1 = {inputs[3]:5.2f} deg, Re = {inputs[0]:.3E}): "
                                               f"Cl = {aero_data['Cl']:7.4f} | Cd = {aero_data['Cd']:.5f} (Cdp = {aero_data['Cdp']:.5f}, Cdf = {aero_data['Cdf']:.5f}) | Cm = {aero_data['Cm']:7.4f} "
                                               f"| L/D = {aero_data['L/D']:8.4f}\n")
            sb = self.text_area.verticalScrollBar()
            sb.setValue(sb.maximum())

            if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
                if self.analysis_graph is None:
                    # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow geometry docking window to be closed
                    if self.dark_mode:
                        bcolor = '#2a2a2b'
                    else:
                        bcolor = 'w'
                    self.analysis_graph = AnalysisGraph(background_color=bcolor)
                    self.dockable_tab_window.add_new_tab_widget(self.analysis_graph.w, "Analysis")
                pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pens[self.n_converged_analyses][0],
                                                                         style=self.pens[self.n_converged_analyses][1]),
                                                            name=str(self.n_analyses))
                pg_plot_handle.setData(aero_data['Cp']['x'], aero_data['Cp']['Cp'])
                # pen = pg.mkPen(color='green')
                self.n_converged_analyses += 1
                self.n_analyses += 1
            else:
                self.n_analyses += 1

    def match_airfoil(self):
        target_airfoil = 'A0'
        match_airfoil(self.mea, target_airfoil, 'sc20010-il')

    # def add_airfoil(self, airfoil: Airfoil = None):
    #     if not airfoil:
    #         airfoil = Airfoil(
    #             base_airfoil_params=BaseAirfoilParams(dx=Param(-0.1), dy=Param(-0.2)))
    #     self.airfoil_graphs.append(AirfoilGraph(airfoil=airfoil, w=self.w, v=self.v))

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
    if len(sys.argv) > 1:
        gui = GUI(sys.argv[1])
    else:
        gui = GUI()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
