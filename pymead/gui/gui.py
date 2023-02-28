import tempfile

from pymead.gui.rename_popup import RenamePopup
from pymead.gui.main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, \
    QWidget, QMenu, QStatusBar, QAction, QGraphicsScene, QGridLayout, QSizePolicy
from PyQt5.QtGui import QIcon, QFont, QFontDatabase, QPainter
from PyQt5.QtCore import QEvent, QObject, Qt, QThreadPool
from PyQt5.QtSvg import QSvgWidget


from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead import RESOURCE_DIR
from pymead.gui.input_dialog import SingleAirfoilViscousDialog, LoadDialog, SaveAsDialog, OptimizationSetupDialog, \
    MultiAirfoilDialog, ColorInputDialog, ExportCoordinatesDialog, ExportControlPointsDialog, AirfoilPlotDialog, \
    AirfoilMatchingDialog, MSESFieldPlotDialog
from pymead.gui.pymeadPColorMeshItem import PymeadPColorMeshItem
from pymead.gui.analysis_graph import AnalysisGraph
from pymead.gui.parameter_tree import MEAParamTree
from pymead.utils.airfoil_matching import match_airfoil
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.dockable_tab_widget import DockableTabWidget
from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.optimization.opt_setup import CustomDisplay, TPAIOPT, SelfIntersectionRepair
from pymead.utils.read_write_files import load_data, save_data
from pymead.analysis.read_aero_data import read_grid_stats_from_mses
from pymead.utils.misc import make_ga_opt_dir
from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.optimization.pop_chrom import Chromosome, Population, CustomGASettings
from pymead.optimization.custom_ga_sampling import CustomGASampling
from pymead.optimization.opt_setup import termination_condition, calculate_warm_start_index, \
    convert_opt_settings_to_param_dict
from pymead.gui.message_box import disp_message_box
from pymead.gui.worker import Worker
from pymead.optimization.opt_callback import PlotAirfoilCallback, ParallelCoordsCallback, OptCallback, \
    DragPlotCallbackXFOIL, CpPlotCallbackXFOIL, DragPlotCallbackMSES, CpPlotCallbackMSES
from pymead.gui.input_dialog import convert_dialog_to_mset_settings, convert_dialog_to_mses_settings, \
    convert_dialog_to_mplot_settings
from pymead.gui.airfoil_statistics import AirfoilStatisticsDialog, AirfoilStatistics
from pymead.gui.custom_graphics_view import CustomGraphicsView
from pymead.gui.file_selection import select_directory
from pymead.utils.dict_recursion import unravel_param_dict_deepcopy
from pymead.core.param import Param

import pymoo.core.population
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.config import Config
from pymoo.core.evaluator import Evaluator
from pymoo.factory import get_reference_directions
from pymoo.core.evaluator import set_cv
from pymead.analysis.calc_aero_data import SVG_PLOTS, SVG_SETTINGS_TR
from pyqtgraph.exporters import CSVExporter, SVGExporter

import pyqtgraph as pg
import numpy as np
import dill
from copy import deepcopy
from functools import partial
from pymead.version import __version__
import benedict
import shutil
import sys
import os
import random


class GUI(QMainWindow):
    def __init__(self, path=None, parent=None):
        # super().__init__(flags=Qt.FramelessWindowHint)
        super().__init__(parent=parent)
        # self.setWindowFlags(Qt.CustomizeWindowHint)
        self.menu_bar = None
        self.path = path
        # single_element_inviscid(np.array([[1, 0], [0, 0], [1, 0]]), 0.0)
        for font_name in ["DejaVuSans", "DejaVuSansMono", "DejaVuSerif"]:
            QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "dejavu-fonts-ttf-2.37", "ttf",
                                                          f"{font_name}.ttf"))
        # QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "cascadia-code", "Cascadia.ttf"))
        # print(QFontDatabase().families())

        self.design_tree = None
        self.dialog = None
        self.save_attempts = 0
        self.opt_settings = None
        self.multi_airfoil_analysis_settings = None
        self.xfoil_settings = None
        self.current_settings_save_file = None
        self.cbar = None
        self.default_field_dir = None
        self.objectives = []
        self.constraints = []
        self.airfoil_name_list = []
        self.analysis_graph = None
        self.opt_airfoil_graph = None
        self.parallel_coords_graph = None
        self.drag_graph = None
        self.Cp_graph = None
        self.geometry_plot_handles = {}
        # self.Mach_contour_widget = None
        # self.grid_widget = None
        # self.finished_optimization = False
        self.opt_airfoil_plot_handles = []
        self.parallel_coords_plot_handles = []
        self.Cp_graph_plot_handles = []
        self.forces_dict = {}
        self.te_thickness_edit_mode = False
        self.dark_mode = False
        self.worker = None
        self.n_analyses = 0
        self.n_converged_analyses = 0
        self.threadpool = QThreadPool().globalInstance()
        self.threadpool.setMaxThreadCount(1)
        self.pens = [('#d4251c', Qt.SolidLine), ('darkorange', Qt.SolidLine), ('gold', Qt.SolidLine),
                     ('limegreen', Qt.SolidLine), ('cyan', Qt.SolidLine), ('mediumpurple', Qt.SolidLine),
                     ('deeppink', Qt.SolidLine), ('#d4251c', Qt.DashLine), ('darkorange', Qt.DashLine),
                     ('gold', Qt.DashLine),
                     ('limegreen', Qt.DashLine), ('cyan', Qt.DashLine), ('mediumpurple', Qt.DashLine),
                     ('deeppink', Qt.DashLine)]
        # self.setFont(QFont("DejaVu Serif"))
        self.setFont(QFont("DejaVu Sans"))

        self.mea = MEA(airfoil_graphs_active=True)
        # self.mea.add_airfoil_graph_to_airfoil(self.mea.airfoils['A0'], 0, None)

        # self.mea.airfoils['A0'].insert_free_point(FreePoint(Param(0.5), Param(0.1), previous_anchor_point='te_1'))
        # self.mea.airfoils['A0'].update()
        # self.airfoil_graphs = [AirfoilGraph(self.mea.airfoils['A0'])]
        self.w = pg.GraphicsLayoutWidget(show=True, size=(1000, 300))
        self.w.setBackground('#2a2a2b')
        self.v = self.w.addPlot()
        self.v.setAspectLocked()
        self.v.hideButtons()
        # self.w = self.mea.airfoils['A0'].airfoil_graph.w
        # self.v = self.mea.airfoils['A0'].airfoil_graph.v
        # internal_geometry_xy = np.loadtxt(os.path.join(DATA_DIR, 'sec_6.txt'))
        # # print(f"geometry = {internal_geometry_xy}")
        # scale_factor = 0.612745
        # x_start = 0.13352022
        # self.internal_geometry = self.v.plot(internal_geometry_xy[:, 0] * scale_factor + x_start,
        #                                      internal_geometry_xy[:, 1] * scale_factor,
        #                                      pen=pg.mkPen(color='orange', width=1))
        # self.airfoil_graphs.append(AirfoilGraph(self.mea.airfoils['A1'], w=self.w, v=self.v))
        self.main_layout = QHBoxLayout()
        self.setStatusBar(QStatusBar(self))
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        # self.mea.airfoils['A0'].airfoil_graph.param_tree = self.param_tree_instance
        # self.mea.airfoils['A0'].airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param(
        #     'Airfoil Parameters')
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
        # self.main_layout_upper.addWidget(MyBar(self))
        # self.main_layout_upper.addWidget(self.main_widget)
        # self.main_widget_upper = QWidget()
        # self.main_widget_upper.setLayout(self.main_layout_upper)
        self.setCentralWidget(self.main_widget)

        # self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

        # self.resize(640, self.titleBar.height() + 480)

        self.set_title_and_icon()
        self.create_menu_bar()
        self.main_icon_toolbar = MainIconToolbar(self)
        if self.main_icon_toolbar.buttons["change-background-color"]["button"].isChecked():
            self.set_dark_mode()
        if self.path is not None:
            self.load_mea_no_dialog(self.path)
        self.output_area_text(f"<font color='#1fbbcc' size='5'>pymead</font> <font size='5'>version</font> "
                              f"<font color='#44e37e' size='5'>{__version__}</font>",
                              mode='html')
        self.output_area_text('\n')
        # self.output_area_text("<font color='#ffffff' size='3'>\n\n</font>", mode='html')
        airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(0.0), dy=Param(0.0)))
        self.add_airfoil(airfoil)
        self.auto_range_geometry()
        self.showMaximized()
        # for dw in self.dockable_tab_window.dock_widgets:
        #     dw.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

    def set_dark_mode(self):
        self.setStyleSheet("background-color: #3e3f40; color: #dce1e6; font-family: DejaVu; font-size: 12px;")
        for dock_widget in self.dockable_tab_window.dock_widgets:
            if hasattr(dock_widget.widget(), 'setBackground'):
                dock_widget.widget().setBackground('#2a2a2b')

    def set_light_mode(self):
        self.setStyleSheet("font-family: DejaVu; font-size: 12px;")
        for dock_widget in self.dockable_tab_window.dock_widgets:
            if hasattr(dock_widget.widget(), 'setBackground'):
                dock_widget.widget().setBackground('w')

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(os.path.dirname(os.getcwd()), 'icons', 'pymead.png')
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        menu_data = load_data('menu.json')

        def recursively_add_menus(menu: dict, menu_bar: QObject):
            for key, val in menu.items():
                if isinstance(val, dict):
                    menu_bar.addMenu(QMenu(key, parent=menu_bar))
                    recursively_add_menus(val, menu_bar.children()[-1])
                else:
                    action = QAction(key, parent=menu_bar)
                    action_parent = action.parent()
                    if isinstance(action_parent, QMenu):
                        action_parent.addAction(action)
                    else:
                        raise ValueError('Attempted to add QAction to an object not of type QMenu')
                    if isinstance(val, list):
                        action.triggered.connect(getattr(self, val[0]))
                        action.setShortcut(val[1])
                    else:
                        action.triggered.connect(getattr(self, val))

        recursively_add_menus(menu_data, self.menu_bar)

    # TODO: add "New" function to menu to allow for delete or save of current MEA and opening of new default MEA

    def save_as_mea(self):
        dialog = SaveAsDialog(self)
        if dialog.exec_():
            self.mea.file_name = dialog.selectedFiles()[0]
            if self.mea.file_name[-5:] != '.jmea':
                self.mea.file_name += '.jmea'
            self.save_mea()
            self.disp_message_box(f"Multi-element airfoil saved as {self.mea.file_name}", message_mode='info')
        else:
            if self.save_attempts > 0:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')

    def save_mea(self):
        if self.mea.file_name is None:
            if self.save_attempts < 1:
                self.save_attempts += 1
                self.save_as_mea()
            else:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')
        else:
            save_data(self.copy_mea(), self.mea.file_name)
            self.save_attempts = 0

    def copy_mea(self):
        output_dict_ = {}
        unravel_param_dict_deepcopy(self.mea.param_dict, output_dict=output_dict_)
        for k, v in output_dict_.items():
            if k != 'Custom':
                output_dict_[k]['anchor_point_order'] = deepcopy(self.mea.airfoils[k].anchor_point_order)
                output_dict_[k]['free_point_order'] = deepcopy(self.mea.airfoils[k].free_point_order)
        output_dict_['file_name'] = self.mea.file_name
        output_dict_['airfoil_graphs_active'] = self.mea.airfoil_graphs_active
        mea_copy = deepcopy(output_dict_)
        return mea_copy

    def load_mea(self):
        dialog = LoadDialog(self)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
        else:
            file_name = None
        if file_name is not None:
            self.load_mea_no_dialog(file_name)

    def auto_range_geometry(self):
        x_data_range, y_data_range = self.mea.get_curve_bounds()
        self.v.getViewBox().setRange(xRange=x_data_range, yRange=y_data_range)

    def import_parameter_list(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            parameter_list = np.loadtxt(file_name).tolist()
            for airfoil in self.mea.airfoils.values():
                airfoil.airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')
            self.mea.update_parameters(parameter_list)
            pass

    def plot_geometry(self):
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            coords = np.loadtxt(file_name)
            geometry_idx = 0
            while True:
                geometry_name = f"Geometry_{geometry_idx}"
                if geometry_name in self.geometry_plot_handles.keys():
                    geometry_idx += 1
                else:
                    default_color = (214, 147, 39)
                    color_dialog = ColorInputDialog(parent=self, default_color=default_color)
                    if color_dialog.exec_():
                        color = color_dialog.color_button_widget.color()
                    else:
                        color = default_color
                    self.geometry_plot_handles[geometry_name] = self.v.plot(pen=pg.mkPen(color=color), lw=1.4)
                    self.geometry_plot_handles[geometry_name].setData(coords[:, 0], coords[:, 1])
                    break

    def clear_field(self):
        for child in self.v.allChildItems():
            if isinstance(child, pg.PColorMeshItem) or isinstance(child, PymeadPColorMeshItem):
                self.v.getViewBox().removeItem(child)
        if self.cbar is not None:
            self.w.removeItem(self.cbar)
            self.cbar = None

    def plot_field(self):

        dlg = MSESFieldPlotDialog(parent=self, default_field_dir=self.default_field_dir)
        if dlg.exec_():
            inputs = dlg.getInputs()
        else:
            return

        self.clear_field()

        for child in self.v.allChildItems():
            if hasattr(child, 'setZValue'):
                child.setZValue(1.0)

        analysis_dir = inputs['analysis_dir']
        vBox = self.v.getViewBox()
        field_file = os.path.join(analysis_dir, f'field.{os.path.split(analysis_dir)[-1]}')
        grid_file = os.path.join(analysis_dir, 'mplot_grid_stats.log')
        if not os.path.exists(field_file):
            self.disp_message_box(message=f"Field file {field_file} not found", message_mode='error')
            return
        if not os.path.exists(grid_file):
            self.disp_message_box(message=f"Grid statistics log {grid_file} not found", message_mode='error')
            return

        self.default_field_dir = analysis_dir

        data = np.loadtxt(field_file, skiprows=2)
        grid = read_grid_stats_from_mses(grid_file)

        with open(field_file, 'r') as f:
            lines = f.readlines()

        n_streamlines = 0
        for line in lines:
            if line == '\n':
                n_streamlines += 1

        n_streamwise_lines = int(data.shape[0] / n_streamlines)

        x = data[:, 0].reshape(n_streamlines, n_streamwise_lines).T
        y = data[:, 1].reshape(n_streamlines, n_streamwise_lines).T

        flow_var_idx = {'M': 7, 'Cp': 8, 'p': 5, 'rho': 4, 'u': 2, 'v': 3, 'q': 6}

        flow_var_label = {'M': 'Mach Number',
                          'Cp': 'Pressure Coefficient',
                          'p': 'Static Pressure (p / p<sub>\u221e</sub>)',
                          'rho': 'Density (\u03c1/\u03c1<sub>\u221e</sub>)',
                          'u': 'Velocity-x (u/V<sub>\u221e</sub>)',
                          'v': 'Velocity-y (v/V<sub>\u221e</sub>)',
                          'q': 'Speed of Sound (q/V<sub>\u221e</sub>)'}

        flow_var = data[:, flow_var_idx[inputs['flow_variable']]].reshape(n_streamlines, n_streamwise_lines).T[:-1, :-1]

        edgecolors = None
        antialiasing = False
        # edgecolors = {'color': 'b', 'width': 1}  # May be uncommented to see edgecolor effect
        # antialiasing = True # May be uncommented to see antialiasing effect
        pcmi = PymeadPColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing,
                                    colorMap=pg.colormap.get('CET-R1'))
        vBox.addItem(pcmi)
        # vBox.addItem(pg.ArrowItem(pxMode=False, headLen=0.01, pos=(0.5, 0.1)))
        vBox.setAspectLocked(True, 1)
        pcmi.setZValue(0)

        pcmi.setData(x, y, flow_var)

        gray_color_mesh_items = []
        for el in range(grid['numel']):
            offset = grid['numel'] - el
            x_gray = x[:, grid['Jside2'][el] - 1 - offset:grid['Jside1'][el] - offset]
            y_gray = y[:, grid['Jside2'][el] - 1 - offset:grid['Jside1'][el] - offset]
            v_gray = np.zeros(shape=x_gray.shape)[:-1, :-1]
            gray_color_item = pg.PColorMeshItem(colorMap=pg.colormap.get('CET-C5s'))
            gray_color_item.setData(x_gray, y_gray, v_gray)
            gray_color_item.setZValue(1)
            gray_color_mesh_items.append(vBox.addItem(gray_color_item))

        for child in self.v.allChildItems():
            if hasattr(child, 'setZValue') and not isinstance(child, pg.PColorMeshItem) and not isinstance(child, PymeadPColorMeshItem):
                child.setZValue(5)

        bar = pg.ColorBarItem(
            values=pcmi.getLevels(),
            colorMap='CET-R1',
            rounding=0.001,
            limits=pcmi.getLevels(),
            orientation='v',
            pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        )
        bar.setLabel(axis='right', text=flow_var_label[inputs['flow_variable']], **{'font-size': '12pt',
                                                                                    'color': '#ffffff'})
        self.cbar = bar
        self.w.addItem(bar)
        pcmi.disableAutoLevels()

        def on_levels_changed(cbar):
            pcmi.setLevels(cbar.levels())

        bar.sigLevelsChanged.connect(on_levels_changed)

    def load_mea_no_dialog(self, file_name):
        self.mea = MEA.generate_from_param_dict(load_data(file_name))
        for a in self.mea.airfoils.values():
            a.update()
        self.v.clear()
        self.param_tree_instance.t.clear()
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, None, w=self.w, v=self.v)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        for a in self.mea.airfoils.values():
            a.airfoil_graph.param_tree = self.param_tree_instance
            a.airfoil_graph.airfoil_parameters = a.airfoil_graph.param_tree.p.param('Airfoil Parameters')
        dben = benedict.benedict(self.mea.param_dict)
        for k in dben.keypaths():
            param = dben[k]
            if isinstance(param, Param):
                if param.mea is None:
                    param.mea = self.mea
                if param.mea.param_tree is None:
                    param.mea.param_tree = self.param_tree_instance
        self.mea.param_tree = self.param_tree_instance
        self.design_tree_widget = self.param_tree_instance.t
        widget0 = self.main_layout.itemAt(0).widget()
        self.main_layout.replaceWidget(widget0, self.design_tree_widget)
        widget0.deleteLater()
        self.auto_range_geometry()

    def add_airfoil(self, airfoil: Airfoil):
        self.mea.te_thickness_edit_mode = self.te_thickness_edit_mode
        self.mea.add_airfoil(airfoil, len(self.mea.airfoils), self.param_tree_instance,
                                    w=self.w, v=self.v)
        self.airfoil_name_list = [k for k in self.mea.airfoils.keys()]
        # self.param_tree_instance.p.child("Analysis").child("Inviscid Cl Calc").setLimits([a.tag for a in self.mea.airfoils.values()])
        self.param_tree_instance.params[-1].add_airfoil(airfoil, len(self.mea.airfoils) - 1)
        for a in self.mea.airfoils.values():
            if a.airfoil_graph.airfoil_parameters is None:
                a.airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')
        airfoil.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed,
                                                                     f"A{len(self.mea.airfoils) - 1}"))

    def disp_message_box(self, message: str, message_mode: str = 'error'):
        disp_message_box(message, self, message_mode=message_mode)

    def output_area_text(self, text: str, mode: str = 'plain'):
        if mode == 'plain':
            self.text_area.insertPlainText(text)
        elif mode == 'html':
            self.text_area.insertHtml(text)
        else:
            raise ValueError('Mode must be \'plain\' or \'html\'')
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def display_airfoil_statistics(self):
        airfoil_stats = AirfoilStatistics(mea=self.mea)
        dialog = AirfoilStatisticsDialog(parent=self, airfoil_stats=airfoil_stats)
        dialog.exec()

    def single_airfoil_inviscid_analysis(self):
        """Inviscid analysis not yet implemented here"""
        pass

    def export_coordinates(self):
        """Airfoil coordinate exporter"""
        dialog = ExportCoordinatesDialog(self)
        if dialog.exec_():
            inputs = dialog.getInputs()
            f_ = os.path.join(inputs['choose_dir'], inputs['file_name'])

            # Determine if output format should be JSON:
            if os.path.splitext(f_) and os.path.splitext(f_)[-1] == '.json':
                json = True
            else:
                json = False

            airfoils = inputs['airfoil_order'].split(',')

            if json:
                coord_dict = {}
                for a in airfoils:
                    airfoil = self.mea.airfoils[a]
                    coords = airfoil.get_coords(body_fixed_csys=False)
                    coord_dict[a] = coords.tolist()
                save_data(coord_dict, f_)
            else:
                with open(f_, 'w') as f:
                    new_line = ""
                    if len(inputs['header']) > 0:
                        new_line = '\n'
                    f.write(f"{inputs['header']}{new_line}")
                    for idx, a in enumerate(airfoils):
                        airfoil = self.mea.airfoils[a]
                        coords = airfoil.get_coords(body_fixed_csys=False)
                        for coord in coords:
                            f.write(f"{coord[0]}{inputs['delimiter']}{coord[1]}\n")
                        if idx < len(airfoils) - 1:
                            f.write(f"{inputs['separator']}")
            self.disp_message_box(f"Airfoil coordinates saved to {f_}", message_mode='info')

    def export_control_points(self):
        dialog = ExportControlPointsDialog(self)
        if dialog.exec_():
            inputs = dialog.getInputs()
            f_ = os.path.join(inputs['choose_dir'], inputs['file_name'])

            airfoils = inputs['airfoil_order'].split(',')

            control_point_dict = {}
            for a in airfoils:
                airfoil = self.mea.airfoils[a]
                control_points = []
                for c in airfoil.curve_list:
                    control_points.append(c.P.tolist())
                control_point_dict[a] = control_points
            save_data(control_point_dict, f_)
            self.disp_message_box(f"Airfoil control points saved to {f_}", message_mode='info')

    def export_nx_macro(self):
        self.mea.write_NX_macro('test_ctrlpts.py', {})

    def single_airfoil_viscous_analysis(self):
        self.dialog = SingleAirfoilViscousDialog(parent=self)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None

        if inputs is not None:
            xfoil_settings = {'Re': inputs['Re']['value'],
                              'Ma': inputs['Ma']['value'],
                              'prescribe': inputs['prescribe']['current_text'],
                              'timeout': inputs['timeout']['value'],
                              'iter': inputs['iter']['value'],
                              'xtr': [inputs['xtr_lower']['value'], inputs['xtr_upper']['value']],
                              'N': inputs['N']['value'],
                              'airfoil_analysis_dir': inputs['airfoil_analysis_dir']['text'],
                              'airfoil_coord_file_name': inputs['airfoil_coord_file_name']['text'],
                              'airfoil': inputs['airfoil']['current_text']}
            if xfoil_settings['prescribe'] == 'Angle of Attack (deg)':
                xfoil_settings['alfa'] = inputs['alfa']['value']
            elif xfoil_settings['prescribe'] == 'Viscous Cl':
                xfoil_settings['Cl'] = inputs['Cl']['value']
            elif xfoil_settings['prescribe'] == 'Inviscid Cl':
                xfoil_settings['CLI'] = inputs['CLI']['value']
            aero_data, _ = calculate_aero_data(xfoil_settings['airfoil_analysis_dir'],
                                               xfoil_settings['airfoil_coord_file_name'], self.mea,
                                               xfoil_settings['airfoil'], 'xfoil',
                                               xfoil_settings, body_fixed_csys=inputs['body_fixed_csys']['state'])
            if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
                self.text_area.insertPlainText(
                    f"[{self.n_analyses:2.0f}] Converged = {aero_data['converged']} | Errored out = "
                    f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}\n")
            else:
                self.text_area.insertPlainText(
                    f"[{self.n_analyses:2.0f}] ({xfoil_settings['airfoil']}, \u03b1 = {aero_data['alf']:.3f}, Re = {xfoil_settings['Re']:.3E}, Ma = {xfoil_settings['Ma']:.3f}): "
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

    def multi_airfoil_analysis_setup(self):

        # First check to make sure MSET, MSES, and MPLOT can be found on system path and marked as executable:
        if shutil.which('mset') is None:
            self.disp_message_box('MSES suite executable \'mset\' not found on system path')
            return
        if shutil.which('mses') is None:
            self.disp_message_box('MSES suite executable \'mses\' not found on system path')
            return
        if shutil.which('mplot') is None:
            self.disp_message_box('MPLOT suite executable \'mplot\' not found on system path')
            return

        self.dialog = MultiAirfoilDialog(parent=self, settings_override=self.multi_airfoil_analysis_settings)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
            self.multi_airfoil_analysis_settings = inputs
        else:
            inputs = None
            self.multi_airfoil_analysis_settings = self.dialog.getInputs()

        if inputs is not None:
            mset_settings = convert_dialog_to_mset_settings(inputs['MSET'])
            mses_settings = convert_dialog_to_mses_settings(inputs['MSES'])
            mses_settings['n_airfoils'] = mset_settings['n_airfoils']
            mplot_settings = convert_dialog_to_mplot_settings(inputs['MPLOT'])
            self.multi_airfoil_analysis(mset_settings, mses_settings, mplot_settings)

    def multi_airfoil_analysis(self, mset_settings: dict, mses_settings: dict,
                               mplot_settings: dict):
        aero_data, _ = calculate_aero_data(mset_settings['airfoil_analysis_dir'],
                                           mset_settings['airfoil_coord_file_name'],
                                           self.mea,
                                           '',
                                           tool='MSES',
                                           export_Cp=True,
                                           mset_settings=mset_settings,
                                           mses_settings=mses_settings,
                                           mplot_settings=mplot_settings)
        if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
            self.disp_message_box("MSES Analysis Failed", message_mode='error')
            self.output_area_text(
                f"<font color='ffffff'>[{self.n_analyses:2.0f}] Converged = {aero_data['converged']} | Errored out = "
                f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}</font>", mode='html')
        else:
            # self.output_area_text('\n')
            self.output_area_text(
                f"<font size='4'>[{self.n_analyses:2.0f}] (Re = {mses_settings['REYNIN']:.3E}, Ma = {mses_settings['MACHIN']:.3f}): "
                f"Cl = {aero_data['Cl']:7.4f} | Cd = {aero_data['Cd']:.5f} | Cm = {aero_data['Cm']:7.4f}</font>", mode='html')
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

        if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
            if self.analysis_graph is None:
                # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow
                # geometry docking window to be closed
                if self.dark_mode:
                    bcolor = '#2a2a2b'
                else:
                    bcolor = 'w'
                self.analysis_graph = AnalysisGraph(background_color=bcolor)
                self.dockable_tab_window.add_new_tab_widget(self.analysis_graph.w, "Analysis")
            pen_idx = self.n_converged_analyses % len(self.pens)
            x_max = self.mea.calculate_max_x_extent()
            for side in aero_data['BL']:
                pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pens[pen_idx][0],
                                                                         style=self.pens[pen_idx][1]),
                                                            name=str(self.n_analyses))
                x = side['x']
                Cp = side['Cp']
                if not isinstance(x, np.ndarray):
                    x = np.array(x)
                if not isinstance(Cp, np.ndarray):
                    Cp = np.array(Cp)
                pg_plot_handle.setData(x[np.where(x <= x_max)[0]], Cp[np.where(x <= x_max)[0]])
            # pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pens[pen_idx][0],
            #                                                          style=self.pens[pen_idx][1]),
            #                                             name=str(self.n_analyses))
            # pg_plot_handle.setData(aero_data['BL'][0]['x'], aero_data['BL'][0]['Cp'])
            # pen = pg.mkPen(color='green')
            self.n_converged_analyses += 1
            self.n_analyses += 1
            for svg_plot in SVG_PLOTS:
                if mplot_settings[SVG_SETTINGS_TR[svg_plot]]:
                    f_name = os.path.join(mset_settings['airfoil_analysis_dir'],
                                          mset_settings['airfoil_coord_file_name'],
                                          f"{svg_plot}.svg")
                    if os.path.exists(f_name):
                        image = QSvgWidget(f_name)
                        graphics_scene = QGraphicsScene()
                        graphics_scene.addWidget(image)
                        view = CustomGraphicsView(graphics_scene, parent=self)
                        view.setRenderHint(QPainter.Antialiasing)
                        Mach_contour_widget = QWidget(self)
                        widget_layout = QGridLayout()
                        Mach_contour_widget.setLayout(widget_layout)
                        widget_layout.addWidget(view, 0, 0, 4, 4)
                        # new_image = QSvgWidget(os.path.join(RESOURCE_DIR, 'sec_34.svg'))
                        # temp_widget.setWidget(new_image)
                        start_counter = 1
                        max_tab_name_search = 1000
                        for idx in range(max_tab_name_search):
                            name = f"{svg_plot}_{start_counter}"
                            if name in self.dockable_tab_window.names:
                                start_counter += 1
                            else:
                                self.dockable_tab_window.add_new_tab_widget(Mach_contour_widget, name)
                                break
        else:
            self.n_analyses += 1

    def match_airfoil(self):
        target_airfoil = 'A0'
        dialog = AirfoilMatchingDialog(self)
        if dialog.exec_():
            airfoil_name = dialog.getInputs()
            res = match_airfoil(self.mea, target_airfoil, airfoil_name)
            if res.success:
                self.mea.update_parameters(res.x)
            msg_mode = 'info' if res.success else 'error'
            self.disp_message_box(message=res.message, message_mode=msg_mode)

    def plot_airfoil_from_airfoiltools(self):
        dialog = AirfoilPlotDialog(self)
        if dialog.exec_():
            airfoil_name = dialog.getInputs()
            airfoil = extract_data_from_airfoiltools(airfoil_name)
            self.v.plot(airfoil[:, 0], airfoil[:, 1], pen=pg.mkPen(color='orange', width=1))

    def setup_optimization(self):
        exit_the_dialog = False
        early_return = False
        param_dict = None
        opt_settings = None
        opt_settings_list = None
        param_dict_list = None
        mea_list = None
        files = None
        mea = None
        dialog = OptimizationSetupDialog(self)
        if dialog.exec_():
            while not exit_the_dialog and not early_return:
                self.opt_settings = dialog.getInputs()

                loop_through_settings = False

                if self.opt_settings['Warm Start/Batch Mode']['batch_mode_active']['state'] in [1, 2]:

                    loop_through_settings = True

                    files = self.opt_settings['Warm Start/Batch Mode']['batch_mode_files']['texts']

                    if files == ['']:
                        self.disp_message_box('The \'Batch Settings Files\' field must be filled because batch mode '
                                              'is selected as active', message_mode='error')
                        exit_the_dialog = True
                        early_return = True
                        continue

                    all_batch_files_valid = True
                    opt_settings_list = []
                    for file in files:
                        if not os.path.exists(file):
                            self.disp_message_box(f'The batch file {file} could not be located', message_mode='error')
                            exit_the_dialog = True
                            early_return = True
                            all_batch_files_valid = False
                            break
                        opt_settings_list.append(load_data(file))
                    if not all_batch_files_valid:
                        continue

                if loop_through_settings:
                    n_settings = len(files)
                else:
                    n_settings = 1

                if not loop_through_settings:
                    opt_settings_list = []
                param_dict_list = []
                mea_list = []

                for settings_idx in range(n_settings):

                    if loop_through_settings:
                        opt_settings = opt_settings_list[settings_idx]
                    else:
                        opt_settings = self.opt_settings

                    param_dict = convert_opt_settings_to_param_dict(opt_settings)

                    if not opt_settings['Warm Start/Batch Mode']['use_current_mea']['state']:
                        mea_dict = self.copy_mea()
                    else:
                        mea_file = opt_settings['Warm Start/Batch Mode']['mea_file']['text']
                        if not os.path.exists(mea_file):
                            self.disp_message_box('JMEA parametrization file not found', message_mode='error')
                            exit_the_dialog = True
                            early_return = True
                            continue
                        else:
                            mea_dict = load_data(mea_file)

                    # Generate the multi-element airfoil from the dictionary
                    mea = MEA.generate_from_param_dict(mea_dict)

                    norm_val_list, _ = mea.extract_parameters()
                    if isinstance(norm_val_list, str):
                        error_message = norm_val_list
                        self.disp_message_box(error_message, message_mode='error')
                        exit_the_dialog = True
                        early_return = True
                        continue

                    param_dict['n_var'] = len(norm_val_list)

                    # Thickness distribution check parameters
                    if opt_settings['Constraints/Validation']['check_thickness_at_points']['state']:
                        param_dict['thickness_dist_file'] = \
                            opt_settings['Constraints/Validation']['thickness_at_points']['text']
                        try:
                            data = np.loadtxt(param_dict['thickness_dist_file'])
                            param_dict['thickness_dist'] = data.tolist()
                        except FileNotFoundError:
                            message = f'Thickness file {param_dict["thickness_dist"]} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        param_dict['thickness_dist'] = None

                    # Internal geometry check parameters
                    if opt_settings['Constraints/Validation']['use_internal_geometry']['state']:
                        param_dict['internal_geometry_file'] = \
                            opt_settings['Constraints/Validation']['internal_geometry']['text']
                        try:
                            data = np.loadtxt(param_dict['internal_geometry_file'])
                            param_dict['internal_point_matrix'] = data.tolist()
                        except FileNotFoundError:
                            message = f'Internal geometry file {param_dict["internal_geometry_file"]} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        param_dict['internal_point_matrix'] = None
                    param_dict['int_geometry_timing'] = opt_settings['Constraints/Validation']['internal_geometry_timing'][
                        'current_text']

                    # External geometry check parameters
                    if opt_settings['Constraints/Validation']['use_external_geometry']['state']:
                        param_dict['external_geometry_file'] = \
                            opt_settings['Constraints/Validation']['external_geometry']['text']
                        try:
                            data = np.loadtxt(param_dict['external_geometry_file'])
                            param_dict['external_point_matrix'] = data.tolist()
                        except FileNotFoundError:
                            message = f'External geometry file {param_dict["external-geometry_file"]} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        param_dict['external_point_matrix'] = None
                    param_dict['ext_geometry_timing'] = opt_settings['Constraints/Validation']['external_geometry_timing'][
                        'current_text']

                    # Warm start parameters
                    if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                        opt_dir = opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text']
                    else:
                        opt_dir = make_ga_opt_dir(opt_settings['Genetic Algorithm']['root_dir']['text'],
                                                  opt_settings['Genetic Algorithm']['opt_dir_name']['text'])

                    param_dict['opt_dir'] = opt_dir

                    name_base = 'ga_airfoil'
                    name = [f"{name_base}_{i}" for i in range(opt_settings['Genetic Algorithm']['n_offspring']['value'])]
                    param_dict['name'] = name

                    for airfoil in mea.airfoils.values():
                        airfoil.airfoil_graphs_active = False
                    mea.airfoil_graphs_active = False
                    base_folder = os.path.join(opt_settings['Genetic Algorithm']['root_dir']['text'],
                                               opt_settings['Genetic Algorithm']['temp_analysis_dir_name']['text'])
                    param_dict['base_folder'] = base_folder
                    if not os.path.exists(base_folder):
                        os.mkdir(base_folder)

                    if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                        param_dict['warm_start_generation'] = calculate_warm_start_index(
                            opt_settings['Warm Start/Batch Mode']['warm_start_generation']['value'], opt_dir)
                    param_dict_save = deepcopy(param_dict)
                    if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
                        save_data(param_dict_save, os.path.join(opt_dir, 'param_dict.json'))
                    else:
                        save_data(param_dict_save, os.path.join(
                            opt_dir, f'param_dict_{param_dict["warm_start_generation"]}.json'))

                    if not loop_through_settings:
                        opt_settings_list = [opt_settings]
                    param_dict_list.append(param_dict)
                    mea_list.append(mea)
                    exit_the_dialog = True
        else:
            return

        if early_return:
            self.setup_optimization()

        if not early_return:
            for (opt_settings, param_dict, mea) in zip(opt_settings_list, param_dict_list, mea_list):
                # The next two lines are just to make sure any calls to the GUI are performed before the optimization
                dialog.inputs = opt_settings
                dialog.setInputs()
                self.run_shape_optimization(param_dict, opt_settings, mea)

    def run_shape_optimization(self, param_dict: dict, opt_settings: dict, mea: MEA):
        self.worker = Worker(self.shape_optimization, param_dict, opt_settings, mea)
        self.worker.signals.progress.connect(self.shape_opt_progress_callback_fn)
        self.worker.signals.result.connect(self.shape_opt_result_callback_fn)
        self.worker.signals.finished.connect(self.shape_opt_finished_callback_fn)
        self.worker.signals.error.connect(self.shape_opt_error_callback_fn)
        self.threadpool.start(self.worker)

    def shape_opt_progress_callback_fn(self, progress_object: object):
        if isinstance(progress_object, OptCallback):
            progress_object.exec_callback()

    def shape_opt_finished_callback_fn(self):
        self.output_area_text("Completed optimization.\n")
        # self.finished_optimization = True

    def shape_opt_result_callback_fn(self, result_object: object):
        self.output_area_text(f"Complete! Result = {result_object}\n")

    def shape_opt_error_callback_fn(self, error_tuple: tuple):
        self.output_area_text(f"Error. Error = {error_tuple}\n")

    def shape_optimization(self, param_dict: dict, opt_settings: dict, mea: MEA, progress_callback):
        Config.show_compile_hint = False
        forces = []
        ref_dirs = get_reference_directions("energy", param_dict['n_obj'], param_dict['n_ref_dirs'],
                                            seed=param_dict['seed'])
        ga_settings = CustomGASettings(population_size=param_dict['n_offsprings'],
                                       mutation_bounds=([-0.002, 0.002]),
                                       mutation_methods=('random-reset', 'random-perturb'),
                                       max_genes_to_mutate=2,
                                       mutation_probability=0.06,
                                       max_mutation_attempts_per_chromosome=500)

        problem = TPAIOPT(n_var=param_dict['n_var'], n_obj=param_dict['n_obj'], n_constr=param_dict['n_constr'],
                          xl=param_dict['xl'], xu=param_dict['xu'], param_dict=param_dict, ga_settings=ga_settings)

        if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
            if param_dict['seed'] is not None:
                np.random.seed(param_dict['seed'])
                random.seed(param_dict['seed'])
            tpaiga2_alg_instance = CustomGASampling(param_dict=problem.param_dict, ga_settings=ga_settings, mea=mea)
            population = Population(problem.param_dict, ga_settings, generation=0,
                                    parents=[tpaiga2_alg_instance.generate_first_parent()],
                                    verbose=param_dict['verbose'], mea=mea)
            population.generate()

            n_subpopulations = 0
            fully_converged_chromosomes = []
            while True:  # "Do while" loop (terminate when enough of chromosomes have fully converged solutions)
                subpopulation = deepcopy(population)
                subpopulation.population = subpopulation.population[param_dict['num_processors'] * n_subpopulations:
                                                                    param_dict['num_processors'] * (
                                                                            n_subpopulations + 1)]

                subpopulation.eval_pop_fitness()

                for chromosome in subpopulation.population:
                    if chromosome.fitness is not None:
                        fully_converged_chromosomes.append(chromosome)

                if len(fully_converged_chromosomes) >= param_dict['population_size']:
                    # Truncate the list of fully converged chromosomes to just the first <population_size> number of
                    # chromosomes:
                    fully_converged_chromosomes = fully_converged_chromosomes[:param_dict['population_size']]
                    break

                n_subpopulations += 1

                if n_subpopulations * (param_dict['num_processors'] + 1) > param_dict['n_offsprings']:
                    raise Exception('Ran out of chromosomes to evaluate in initial population generation')

            new_X = None
            J = None
            G = None

            for chromosome in fully_converged_chromosomes:
                if chromosome.fitness is not None:  # This statement should always pass, but shown here for clarity
                    forces.append(chromosome.forces)
                    if new_X is None:
                        if param_dict['n_offsprings'] > 1:
                            new_X = np.array([chromosome.genes])
                        else:
                            new_X = np.array(chromosome.genes)
                    else:
                        new_X = np.row_stack((new_X, np.array(chromosome.genes)))
                    for objective in self.objectives:
                        objective.update(chromosome.forces)
                    for constraint in self.constraints:
                        constraint.update(chromosome.forces)
                    if J is None:
                        J = np.array([obj.value for obj in self.objectives])
                    else:
                        J = np.row_stack((J, np.array([obj.value for obj in self.objectives])))
                    if len(self.constraints) > 0:
                        if G is None:
                            G = np.array([constraint.value for constraint in self.constraints])
                        else:
                            G = np.row_stack((G, np.array([
                                constraint.value for constraint in self.constraints])))
            pop_initial = pymoo.core.population.Population.new("X", new_X)
            # objectives
            pop_initial.set("F", J)
            if len(self.constraints) > 0:
                if G is not None:
                    pop_initial.set("G", G)
            set_cv(pop_initial)
            for individual in pop_initial:
                individual.evaluated = {"F", "G", "CV", "feasible"}
            Evaluator(skip_already_evaluated=True).eval(problem, pop_initial)

            algorithm = UNSGA3(ref_dirs=ref_dirs, sampling=pop_initial, repair=SelfIntersectionRepair(mea=mea),
                               n_offsprings=param_dict['n_offsprings'],
                               crossover=SimulatedBinaryCrossover(eta=param_dict['eta_crossover']),
                               mutation=PolynomialMutation(eta=param_dict['eta_mutation']))

            termination = termination_condition(param_dict)

            display = CustomDisplay()

            # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
            algorithm.setup(problem, termination, display=display, seed=param_dict['seed'], verbose=True,
                            save_history=False)  # Changed save_history to False in order to help prevent paging errors
            # (running out of RAM because each generation of the algorithm stores the entire algorithm history)

            save_data(algorithm, os.path.join(param_dict['opt_dir'], 'algorithm_gen_0.pkl'))

            # np.save('checkpoint', algorithm)
            # until the algorithm has no terminated
            n_generation = 0
        else:
            warm_start_index = param_dict['warm_start_generation']
            n_generation = warm_start_index
            algorithm = load_data(os.path.join(opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text'],
                                               f'algorithm_gen_{warm_start_index}.pkl'))
            term = deepcopy(algorithm.termination.terminations)
            term = list(term)
            term[0].n_max_gen = param_dict['n_max_gen']
            term = tuple(term)
            algorithm.termination.terminations = term
            algorithm.has_terminated = False

        while algorithm.has_next():

            pop = algorithm.ask()

            n_generation += 1

            if n_generation > 1:

                forces = []

                # evaluate (objective function value arrays must be numpy column vectors)
                X = pop.get("X")
                # print(f"{len(X) = }")
                new_X = None
                J = None
                G = None
                n_infeasible_solutions = 0
                search_for_feasible_idx = 0
                while True:
                    gene_matrix = []
                    feasible_indices = []
                    while True:
                        if X[search_for_feasible_idx, 0] != 9999:
                            gene_matrix.append(X[search_for_feasible_idx, :].tolist())
                            feasible_indices.append(search_for_feasible_idx)
                        else:
                            n_infeasible_solutions += 1
                        search_for_feasible_idx += 1
                        if len(gene_matrix) == param_dict['num_processors']:
                            break
                    population = [Chromosome(problem.param_dict, ga_settings=ga_settings, category=None,
                                             generation=n_generation,
                                             population_idx=feasible_indices[idx + len(feasible_indices)
                                                                             - param_dict['num_processors']],
                                             genes=gene_list, verbose=param_dict['verbose'],
                                             mea=mea)
                                  for idx, gene_list in enumerate(gene_matrix)]
                    pop_obj = Population(problem.param_dict, ga_settings=ga_settings, generation=n_generation,
                                         parents=population, verbose=param_dict['verbose'], mea=mea)
                    pop_obj.population = population
                    for chromosome in pop_obj.population:
                        chromosome.generate()
                    pop_obj.eval_pop_fitness()
                    for idx, chromosome in enumerate(pop_obj.population):
                        if chromosome.fitness is not None:
                            forces.append(chromosome.forces)
                            if new_X is None:
                                if param_dict['n_offsprings'] > 1:
                                    new_X = np.array(chromosome.genes)
                                else:
                                    new_X = np.array([chromosome.genes])
                            else:
                                new_X = np.row_stack((new_X, np.array(chromosome.genes)))
                            for objective in self.objectives:
                                objective.update(chromosome.forces)
                            for constraint in self.constraints:
                                constraint.update(chromosome.forces)
                            if J is None:
                                J = np.array([obj.value for obj in self.objectives])
                            else:
                                J = np.row_stack((J, np.array([obj.value for obj in self.objectives])))
                            if len(self.constraints) > 0:
                                if G is None:
                                    G = np.array([constraint.value for constraint in self.constraints])
                                else:
                                    G = np.row_stack((G, np.array([
                                        constraint.value for constraint in self.constraints])))
                    algorithm.evaluator.n_eval += param_dict['num_processors']
                    # print(f"{J = }")
                    # print(f"{J[:, 0] = }")
                    # print(f"{(J[:, 0] < 1000.0).sum() = }")
                    # print(f"{param_dict = }")
                    # print(f"{param_dict['population_size'] = }")
                    if J is None:
                        population_full = False
                    else:
                        if J.ndim > 1:
                            population_full = (J[:, 0] < 1000.0).sum() >= param_dict['population_size']
                        else:
                            population_full = (J[:] < 1000.0).sum() >= param_dict['population_size']
                    if population_full:
                        break
                # Set the objective function values of the remaining individuals to 1000.0
                for idx in range(search_for_feasible_idx, len(X)):
                    new_X = np.row_stack([new_X, X[idx, :]])
                    # f1 = np.append(f1, np.array([1000.0]))
                    # f2 = np.append(f2, np.array([1000.0]))
                    J = np.row_stack((J, 1000.0 * np.ones(param_dict['n_obj'])))
                    if len(self.constraints) > 0:
                        G = np.row_stack((G, 1000.0 * np.ones(param_dict['n_constr'])))
                # new_X = np.append(new_X, 9999 * np.ones(shape=(n_infeasible_solutions, param_dict['n_var'])),
                #                   axis=0)
                # print(f"now 2 len J = {len(J)}, len G = {len(G)}, len X = {len(new_X)}")
                # for idx in range(n_infeasible_solutions):
                #     # f1 = np.append(f1, np.array([1000.0]))
                #     # f2 = np.append(f2, np.array([1000.0]))
                #     J = np.row_stack((J, 1000.0 * np.ones(param_dict['n_obj'])))
                #     if len(self.constraints) > 0:
                #         G = np.row_stack((G, 1000.0 * np.ones(param_dict['n_constr'])))

                for idx in range(param_dict['n_offsprings'] - len(new_X)):
                    # f1 = np.append(f1, np.array([1000.0]))
                    # f2 = np.append(f2, np.array([1000.0]))
                    new_X = np.row_stack((new_X, 9999 * np.ones(param_dict['n_var'])))
                    J = np.row_stack((J, 1000.0 * np.ones(param_dict['n_obj'])))
                    if len(self.constraints) > 0:
                        G = np.row_stack((G, 1000.0 * np.ones(param_dict['n_constr'])))

                pop.set("X", new_X)

                # objectives
                pop.set("F", J)

                # for constraints
                if len(self.constraints) > 0:
                    pop.set("G", G)

                # this line is necessary to set the CV and feasbility status - even for unconstrained
                set_cv(pop)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            progress_callback.emit(algorithm.display.progress_dict)

            if self.dark_mode:
                bcolor = '#2a2a2b'
            else:
                bcolor = 'w'

            if len(self.objectives) == 1:
                if n_generation > 1:
                    X = algorithm.opt.get("X")[0]
                else:
                    X = algorithm.pop.get("X")[0, :]

            best_in_previous_generation = False
            forces_index = 0
            try:
                forces_index = np.where((new_X == X).all(axis=1))[0][0]
            except IndexError:
                best_in_previous_generation = True

            if best_in_previous_generation:
                for k, v in self.forces_dict.items():
                    if k not in self.forces_dict.keys():
                        self.forces_dict[k] = []
                    self.forces_dict[k].append(v[-1])
            else:
                best_forces = forces[forces_index]
                for k, v in best_forces.items():
                    if param_dict['tool'] in ['xfoil', 'XFOIL', 'mses', 'MSES', 'Mses']:
                        if k not in ['converged', 'timed_out', 'errored_out']:
                            if k not in self.forces_dict.keys():
                                self.forces_dict[k] = []
                            self.forces_dict[k].append(v)

            progress_callback.emit(PlotAirfoilCallback(parent=self, mea=mea, X=X.tolist(), background_color=bcolor))
            # TODO: opt airfoils not plotting correctly
            progress_callback.emit(ParallelCoordsCallback(parent=self, mea=mea, X=X.tolist(), background_color=bcolor))
            if param_dict['tool'] == 'XFOIL':
                progress_callback.emit(CpPlotCallbackXFOIL(parent=self, background_color=bcolor))
                progress_callback.emit(DragPlotCallbackXFOIL(parent=self, background_color=bcolor))
            elif param_dict['tool'] == 'MSES':
                progress_callback.emit(CpPlotCallbackMSES(parent=self, background_color=bcolor))
                progress_callback.emit(DragPlotCallbackMSES(parent=self, background_color=bcolor))
                pass

            # do same more things, printing, logging, storing or even modifying the algorithm object
            if n_generation % param_dict['algorithm_save_frequency'] == 0:
                save_data(algorithm, os.path.join(param_dict['opt_dir'], f'algorithm_gen_{n_generation}.pkl'))

        # obtain the result objective from the algorithm
        res = algorithm.result()
        save_data(res, os.path.join(param_dict['opt_dir'], 'res.pkl'))
        save_data(self.forces_dict, os.path.join(param_dict['opt_dir'], 'force_history.json'))
        np.savetxt(os.path.join(param_dict['opt_dir'], 'opt_X.dat'), res.X)
        # self.save_opt_plots(param_dict['opt_dir'])  # not working at the moment

    def save_opt_plots(self, opt_dir: str):
        # Not working at the moment
        opt_plots = {
            'opt_airfoil_graph': 'opt_airfoil',
            'parallel_coords_graph': 'parallel_coordinates',
            'drag_graph': 'drag',
            'Cp_graph': 'Cp'
        }
        for opt_plot in opt_plots.keys():
            if not hasattr(self, opt_plot):
                raise AttributeError(f'No attribute \'{opt_plot}\' found in the GUI')
            attr = getattr(self, opt_plot)
            if attr is not None:
                view = attr.v.getViewBox()
                view.autoRange()
                svg_exporter = SVGExporter(item=attr.v)
                csv_exporter = CSVExporter(item=attr.v)
                svg_exporter.export(os.path.join(opt_dir, f"{opt_plots[opt_plot]}.svg"))
                csv_exporter.export(os.path.join(opt_dir, f"{opt_plots[opt_plot]}.csv"))

    def toggle_full_screen(self):
        if not self.isMaximized():
            self.showMaximized()

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
    app.processEvents()
    app.setStyle('Fusion')
    if len(sys.argv) > 1:
        gui = GUI(sys.argv[1])
    else:
        gui = GUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
