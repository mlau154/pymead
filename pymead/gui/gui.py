import typing
from threading import Thread

import pyqtgraph as pg
import numpy as np
from copy import deepcopy
from functools import partial
import itertools
import benedict
import shutil
import sys
import os
from collections import namedtuple
import multiprocessing as mp

from pymoo.factory import get_decomposition

from pymead.gui.rename_popup import RenamePopup
from pymead.gui.main_icon_toolbar import MainIconToolbar

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, \
    QWidget, QMenu, QStatusBar, QAction, QGraphicsScene, QGridLayout, QProgressBar
from PyQt5.QtGui import QIcon, QFont, QFontDatabase, QPainter, QCloseEvent, QTextCursor
from PyQt5.QtCore import QEvent, QObject, Qt, QThreadPool
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import pyqtSlot

from pymead.version import __version__
from pymead.utils.version_check import using_latest
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.transformation import Transformation3D
from pymead.gui.concurrency import CPUBoundProcess
from pymead.optimization.shape_optimization import shape_optimization as shape_optimization_static
from pymead import RESOURCE_DIR
from pymead.gui.input_dialog import LoadDialog, SaveAsDialog, OptimizationSetupDialog, \
    MultiAirfoilDialog, ColorInputDialog, ExportCoordinatesDialog, ExportControlPointsDialog, AirfoilPlotDialog, \
    AirfoilMatchingDialog, MSESFieldPlotDialog, ExportIGESDialog, XFOILDialog, NewMEADialog, EditBoundsDialog, \
    ExitDialog, ScreenshotDialog, LoadAirfoilAlgFile
from pymead.gui.pymeadPColorMeshItem import PymeadPColorMeshItem
from pymead.gui.analysis_graph import AnalysisGraph
from pymead.gui.parameter_tree import MEAParamTree
from pymead.plugins.IGES.curves import BezierIGES
from pymead.plugins.IGES.iges_generator import IGESGenerator
from pymead.utils.airfoil_matching import match_airfoil
from pymead.optimization.opt_setup import read_stencil_from_array, convert_opt_settings_to_param_dict
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.dockable_tab_widget import DockableTabWidget
from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.misc import count_func_strs
from pymead.analysis.read_aero_data import read_grid_stats_from_mses, read_field_from_mses, \
    read_streamline_grid_from_mses
from pymead.analysis.read_aero_data import flow_var_idx
from pymead.post.mses_field import flow_var_label
from pymead.utils.misc import make_ga_opt_dir
from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.optimization.opt_setup import calculate_warm_start_index
from pymead.gui.message_box import disp_message_box
from pymead.optimization.opt_callback import PlotAirfoilCallback, ParallelCoordsCallback, OptCallback, \
    DragPlotCallbackXFOIL, CpPlotCallbackXFOIL, DragPlotCallbackMSES, CpPlotCallbackMSES, TextCallback
from pymead.gui.input_dialog import convert_dialog_to_mset_settings, convert_dialog_to_mses_settings, \
    convert_dialog_to_mplot_settings
from pymead.gui.airfoil_statistics import AirfoilStatisticsDialog, AirfoilStatistics
from pymead.gui.custom_graphics_view import CustomGraphicsView
from pymead.gui.help_browser import HelpBrowserWindow
from pymead.core.param import Param
from pymead import ICON_DIR, GUI_SETTINGS_DIR, GUI_THEMES_DIR, GUI_DEFAULT_AIRFOIL_DIR, q_settings
from pymead.analysis.calc_aero_data import SVG_PLOTS, SVG_SETTINGS_TR
from pyqtgraph.exporters import CSVExporter, SVGExporter


class GUI(QMainWindow):
    def __init__(self, path=None, parent=None):
        # super().__init__(flags=Qt.FramelessWindowHint)
        super().__init__(parent=parent)
        # self.setWindowFlags(Qt.CustomizeWindowHint)
        print(f"Running GUI with {os.getpid() = }")
        self.pool = None
        self.current_opt_folder = None

        # Set up optimization process (might want to also do this with analysis)
        self.cpu_bound_process = None
        self.opt_thread = None
        self.shape_opt_process = None

        self.closer = None  # This is a string of equal signs used to close out the optimization text progress
        self.menu_bar = None
        self.path = path
        # single_element_inviscid(np.array([[1, 0], [0, 0], [1, 0]]), 0.0)
        for font_name in ["DejaVuSans", "DejaVuSansMono", "DejaVuSerif"]:
            QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "dejavu-fonts-ttf-2.37", "ttf",
                                                          f"{font_name}.ttf"))
        # QFontDatabase.addApplicationFont(os.path.join(RESOURCE_DIR, "cascadia-code", "Cascadia.ttf"))
        # print(QFontDatabase().families())
        self.themes = {
            "dark": load_data(os.path.join(GUI_THEMES_DIR, "dark_theme.json")),
            "light": load_data(os.path.join(GUI_THEMES_DIR, "light_theme.json")),
        }

        self.design_tree = None
        self.dialog = None
        self.save_attempts = 0
        self.opt_settings = None
        self.multi_airfoil_analysis_settings = None
        self.xfoil_settings = None
        self.current_settings_save_file = None
        self.current_theme = "dark"
        self.cbar = None
        self.cbar_label_attrs = None
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
        self.worker = None
        self.n_analyses = 0
        self.n_converged_analyses = 0
        self.threadpool = QThreadPool().globalInstance()
        self.threadpool.setMaxThreadCount(4)
        self.pens = [('#d4251c', Qt.SolidLine), ('darkorange', Qt.SolidLine), ('gold', Qt.SolidLine),
                     ('limegreen', Qt.SolidLine), ('cyan', Qt.SolidLine), ('mediumpurple', Qt.SolidLine),
                     ('deeppink', Qt.SolidLine), ('#d4251c', Qt.DashLine), ('darkorange', Qt.DashLine),
                     ('gold', Qt.DashLine),
                     ('limegreen', Qt.DashLine), ('cyan', Qt.DashLine), ('mediumpurple', Qt.DashLine),
                     ('deeppink', Qt.DashLine)]
        # self.setFont(QFont("DejaVu Serif"))
        self.setFont(QFont("DejaVu Sans"))

        self.mea = MEA(airfoil_graphs_active=True)
        self.w = pg.GraphicsLayoutWidget(show=True, size=(1000, 300))
        self.w.setBackground('#2a2a2b')
        self.v = self.w.addPlot()
        self.v.setAspectLocked()
        self.v.hideButtons()
        self.main_layout = QHBoxLayout()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self)
        self.design_tree_widget = self.param_tree_instance.t
        self.text_area = ConsoleTextArea()
        self.right_widget_layout = QVBoxLayout()
        self.dockable_tab_window = DockableTabWidget(self)
        self.dockable_tab_window.add_new_tab_widget(self.w, "Geometry")
        self.dockable_tab_window.tab_closed.connect(self.on_tab_closed)
        self.right_widget_layout.addWidget(self.dockable_tab_window)
        self.right_widget_layout.addWidget(self.text_area)
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_widget_layout)
        self.main_layout.addWidget(self.design_tree_widget, 1)
        self.main_layout.addWidget(self.right_widget, 3)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.set_title_and_icon()
        self.create_menu_bar()
        self.main_icon_toolbar = MainIconToolbar(self)

        if q_settings.contains("dark_theme_checked"):
            theme = "dark" if q_settings.value("dark_theme_checked") else "light"
        else:
            theme = "dark"
            q_settings.setValue("dark_theme_checked", 2)

        if theme == "dark":
            self.set_theme("dark")
            self.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(True)
        elif theme == "light":
            self.set_theme("light")
            self.main_icon_toolbar.buttons["change-background-color"]["button"].setChecked(False)
        else:
            raise ValueError(f"Current theme options are 'dark' and 'light'. Theme chosen was {theme}")

        self.output_area_text(f"<font color='#1fbbcc' size='5'>pymead</font> <font size='5'>version</font> "
                              f"<font color='#44e37e' size='5'>{__version__}</font>",
                              mode='html')
        self.output_area_text(
            f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style></head><body><p><font size='4'>&#8203;</font></p></body>",
            mode="html")
        self.output_area_text('\n\n')
        # self.output_area_text("<font color='#ffffff' size='3'>\n\n</font>", mode='html')
        airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(0.0), dy=Param(0.0)))
        self.add_airfoil(airfoil)
        self.auto_range_geometry()
        self.statusBar().clearMessage()
        self.progress_bar = QProgressBar(parent=self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet('''QProgressBar {border: 2px solid; border-color: #8E9091;} 
        QProgressBar::chunk {background-color: #6495ED; width: 10px; margin: 0.5px;}
        ''')
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        # self.showMaximized()
        # for dw in self.dockable_tab_window.dock_widgets:
        #     dw.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        # Load the airfoil system from the system argument variable if necessary
        self.mea_start_dict = None
        if self.path is not None:
            self.load_mea_no_dialog(self.path)
        else:
            self.mea_start_dict = self.copy_mea_dict()

        # Check if we are using the most recent release of pymead (notify if not)
        self.check_for_new_version()

    def check_for_new_version(self):
        using_latest_res, latest_ver, current_ver = using_latest()
        if not using_latest_res:
            self.disp_message_box(f"A newer version of pymead ({latest_ver}) is available for download at "
                                  f"<a href='https://github.com/mlau154/pymead/releases' style='color:#45C5E6;'>"
                                  f"https://github.com/mlau154/pymead/releases</a>", message_mode="info",
                                  rich_text=True)

    def closeEvent(self, a0) -> None:
        """
        Close Event handling for the GUI, allowing changes to be saved before exiting the program.

        Parameters
        ==========
        a0: QCloseEvent
            Qt CloseEvent object
        """
        if self.mea_start_dict != self.copy_mea_dict():  # Only run this code if changes have been made
            save_dialog = NewMEADialog(parent=self)
            exit_dialog = ExitDialog(parent=self)
            while True:
                if save_dialog.exec_():  # If "Yes" to "Save Changes,"
                    if save_dialog.save_successful:  # If the changes were saved successfully, close the program.
                        return
                    else:
                        if exit_dialog.exec_():  # Otherwise, If "Yes" to "Exit the Program Anyway," close the program.
                            return
                        else:
                            a0.ignore()
                            return
                else:  # If "Cancel" to "Save Changes," end the CloseEvent and keep the program running.
                    a0.ignore()
                    return

    def on_tab_closed(self, name: str, event: QCloseEvent):
        if name == "Geometry":
            event.ignore()  # Do not allow user to close the geometry window
        elif name == "Analysis":
            self.analysis_graph = None
            self.n_converged_analyses = 0
        elif name == "Opt. Airfoil":
            self.opt_airfoil_graph = None
            self.opt_airfoil_plot_handles = []
        elif name == "Drag":
            self.drag_graph = None
        elif name == "Parallel Coordinates":
            self.parallel_coords_graph = None
            self.parallel_coords_plot_handles = []
        elif name == "Cp":
            self.Cp_graph = None
            self.Cp_graph_plot_handles = []

    @pyqtSlot(str)
    def setStatusBarText(self, message: str):
        self.statusBar().showMessage(message)

    def set_theme(self, theme_name: str):
        self.current_theme = theme_name
        theme = self.themes[theme_name]
        self.setStyleSheet(f"""background-color: {theme['background-color']}; 
                           color: {theme['main-color']};
                           font-family: {theme['font-family']}; font-size: {theme['font-size']};
                           """
                           )
        self.menuBar().setStyleSheet(f"""
                           QMenuBar {{ background-color: {theme['menu-background-color']}; 
                            }}
                            QMenuBar::item:selected {{ background: {theme['menu-item-selected-color']} }}
                           QMenu {{ background-color: {theme['menu-background-color']}; 
                           color: {theme['menu-main-color']};}} 
                           QMenu::item:selected {{ background-color: {theme['menu-item-selected-color']}; }}
                    """)
        for dock_widget in self.dockable_tab_window.dock_widgets:
            if hasattr(dock_widget.widget(), 'setBackground'):
                dock_widget.widget().setBackground(theme["dock-widget-background-color"])
        if self.cbar is not None and self.cbar_label_attrs is not None:
            self.cbar_label_attrs['color'] = theme["cbar-color"]
            self.cbar.setLabel(**self.cbar_label_attrs)
        if self.analysis_graph is not None:
            self.analysis_graph.set_background(theme["graph-background-color"])
        if self.param_tree_instance is not None:
            self.param_tree_instance.set_theme(theme)

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(ICON_DIR, 'pymead-logo.png')
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        menu_data = load_data(os.path.join(GUI_SETTINGS_DIR, "menu.json"))

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

    def take_screenshot(self):

        if hasattr(self.dockable_tab_window, "current_dock_widget"):
            analysis_id = self.dockable_tab_window.current_dock_widget.winId()
        else:
            analysis_id = self.dockable_tab_window.dock_widgets[-1].winId()

        id_dict = {
            "Full Window": self.winId(),
            "Parameter Tree": self.param_tree_instance.t.winId(),
            "Geometry": self.dockable_tab_window.dock_widgets[0].winId(),
            "Analysis": analysis_id,
            "Console": self.text_area.winId()
        }

        dialog = ScreenshotDialog(self)
        if dialog.exec_():
            inputs = dialog.getInputs()

            # Take the screenshot
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(id_dict[inputs["window"]])

            # Handle improper directory names and file extensions
            file_path_split = os.path.split(inputs['image_file'])
            dir_name = file_path_split[0]
            file_name = file_path_split[1]
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1]
            if file_ext not in [".jpg", ".jpeg"]:
                file_ext = ".jpg"

            final_file_name = os.path.join(dir_name, file_name_no_ext + file_ext)

            if os.path.isdir(dir_name):
                screenshot.save(final_file_name, "jpg")  # Save the screenshot
                self.disp_message_box(f"{inputs['window']} window screenshot saved to {final_file_name}",
                                      message_mode="info")
            else:
                self.disp_message_box(f"Directory {dir_name} for"
                                      f"file {file_name} not found")

    def save_as_mea(self):
        dialog = SaveAsDialog(self)
        if dialog.exec_():
            self.mea.file_name = dialog.selectedFiles()[0]
            if self.mea.file_name[-5:] != '.jmea':
                self.mea.file_name += '.jmea'
            self.save_mea()
            self.setWindowTitle(f"pymead - {os.path.split(self.mea.file_name)[-1]}")
            self.disp_message_box(f"Multi-element airfoil saved as {self.mea.file_name}", message_mode='info')
            return True
        else:
            if self.save_attempts > 0:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')
            return False

    def save_mea(self):
        if self.mea.file_name is None:
            if self.save_attempts < 1:
                self.save_attempts += 1
                return self.save_as_mea()
            else:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')
                return False
        else:
            save_data(self.mea.copy_as_param_dict(), self.mea.file_name)
            self.setWindowTitle(f"pymead - {os.path.split(self.mea.file_name)[-1]}")
            self.save_attempts = 0
            return True

    def copy_mea(self):
        return self.mea.deepcopy()

    def copy_mea_dict(self, deactivate_airfoil_graphs: bool = False):
        return self.mea.copy_as_param_dict(deactivate_airfoil_graphs=deactivate_airfoil_graphs)

    def load_mea(self):

        if self.mea_start_dict is not None:
            if self.mea_start_dict != self.copy_mea_dict():
                save_dialog = NewMEADialog(parent=self, message="Airfoil has changes. Save?")
                exit_dialog = ExitDialog(parent=self, window_title="Load anyway?",
                                         message="Airfoil not saved.\nAre you sure you want to load a new one?")
                while True:
                    if save_dialog.exec_():  # If "Yes" to "Save Changes,"
                        if save_dialog.save_successful:  # If the changes were saved successfully, close the program.
                            break
                        else:
                            if exit_dialog.exec_():  # Otherwise, If "Yes" to "Exit the Program Anyway," close the program.
                                break
                        if save_dialog.reject_changes:  # If "No" to "Save Changes," do not load an MEA.
                            return
                    else:  # If "Cancel" to "Save Changes," do not load an MEA
                        return

        dialog = LoadDialog(self, settings_var="jmea_default_open_location")

        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            file_name_parent_dir = os.path.dirname(file_name)
            q_settings.setValue(dialog.settings_var, file_name_parent_dir)
        else:
            file_name = None
        if file_name is not None:
            self.load_mea_no_dialog(file_name)
            self.setWindowTitle(f"pymead - {os.path.split(self.mea.file_name)[-1]}")

    def new_mea(self):
        dialog = NewMEADialog(self)
        if dialog.exec_():
            self.load_mea_no_dialog(os.path.join(GUI_DEFAULT_AIRFOIL_DIR, "default_airfoil.jmea"))
            self.mea.file_name = None
            self.setWindowTitle(f"pymead")

    def get_grandchild(self, full_param_name: str):
        child_list = full_param_name.split(".")
        current_param = self.param_tree_instance.p.child("Airfoil Parameters").child(child_list[0])
        for idx in range(1, len(child_list) - 1):
            current_param = current_param.child(child_list[idx])
        if child_list[0] == "Custom":
            return current_param.child(child_list[-1])
        else:
            return current_param.child(full_param_name)

    def edit_bounds(self):
        jmea_dict = self.mea.copy_as_param_dict(deactivate_airfoil_graphs=True)
        bv_dialog = EditBoundsDialog(parent=self, jmea_dict=jmea_dict)
        if bv_dialog.exec_():
            data_to_modify = bv_dialog.bv_table.compare_data()
            for k, data in data_to_modify.items():
                p = self.get_grandchild(k)
                p.airfoil_param.bounds = data["bounds"]
                p.setValue(data["value"])
                p.airfoil_param.active = data["active"]
                if isinstance(data["value"], list):
                    p.setReadonly(not data["active"][0])
                else:
                    p.setReadonly(not data["active"])
                if data["eq"] is not None:
                    if not p.hasChildren():
                        self.param_tree_instance.add_equation_box(p, data["eq"])
                        self.param_tree_instance.update_equation(p.child("Equation Definition"), data["eq"])
                    else:
                        self.param_tree_instance.update_equation(p.child("Equation Definition"), data["eq"])
                else:
                    if not p.hasChildren():
                        pass
                    else:
                        p.airfoil_param.remove_func()
                        p.setReadonly(False)
                        p.child("Equation Definition").remove()

    def auto_range_geometry(self):
        x_data_range, y_data_range = self.mea.get_curve_bounds()
        self.v.getViewBox().setRange(xRange=x_data_range, yRange=y_data_range)

    def update_airfoil_parameters_from_vector(self, param_vec: np.ndarray):
        for airfoil in self.mea.airfoils.values():
            airfoil.airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')

        try:
            self.mea.update_parameters(param_vec)
        except:
            self.disp_message_box("Could not load parameters into airfoil. Check that the current airfoil system"
                                  " displayed matches the one used in the optimization.")
            return

        self.param_tree_instance.plot_change_recursive(
            self.param_tree_instance.p.param('Airfoil Parameters').child('Custom').children())

    def import_parameter_list(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, settings_var="parameter_list_default_open_location", file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            q_settings.setValue(dialog.settings_var, os.path.dirname(file_name))
            param_vec = np.loadtxt(file_name).tolist()
            self.update_airfoil_parameters_from_vector(param_vec)

    def import_algorithm_pkl_file(self):
        dialog = LoadAirfoilAlgFile(self)
        if dialog.exec_():
            inputs = dialog.getInputs()

            try:
                alg = load_data(inputs["pkl_file"])
            except:
                self.disp_message_box("Could not load .pkl file. Check that the file selected is of the form"
                                      " algorithm_gen_XX.pkl.")
                return

            try:
                X = alg.opt.get("X")
            except AttributeError:
                self.disp_message_box("Algorithm file not recognized. Check that the file selected is of the form"
                                      " algorithm_gen_XX.pkl.")
                return

            if X.shape[0] == 1:  # If single-objective:
                x = X[0, :]
            elif X.shape[0] == 0:  # If the optimization result is empty:
                self.disp_message_box("Empty optimization result")
                return
            else:  # If multi-objective
                if inputs["use_index"]:
                    x = X[inputs["index"], :]
                elif inputs["use_weights"]:
                    F = alg.opt.get("F")
                    decomp = get_decomposition("asf")

                    if len(inputs["weights"]) != F.shape[0]:
                        self.disp_message_box(f"Length of the requested weight list ({len(inputs['weights'])}) does"
                                              f" not match the number of objective functions ({F.shape[0]})")
                        return

                    IDX = decomp.do(F, inputs["weights"]).argmin()
                    x = X[IDX, :]
                else:
                    raise ValueError("Either 'index' or 'weights' must be selected in the dialog")

            self.update_airfoil_parameters_from_vector(x)

    def export_parameter_list(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = SaveAsDialog(self, file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            parameter_list, _ = self.mea.extract_parameters()
            np.savetxt(file_name, np.array(parameter_list))

    def plot_geometry(self):
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, settings_var="geometry_plot_default_open_location", file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            q_settings.setValue(dialog.settings_var, os.path.dirname(file_name))
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
            self.param_tree_instance.p.child("Plot Handles").add_plot_handle(geometry_name, color)

    def clear_geometry(self, name: str):
        self.geometry_plot_handles[name].clear()
        self.geometry_plot_handles.pop(name)
        self.v.update()

    def change_geometry_name(self, name: str, new_names: typing.List[str]):
        old_name = next(iter(set([k for k in self.geometry_plot_handles.keys()]) - set(new_names)))
        temp_handle = self.geometry_plot_handles[old_name]
        self.geometry_plot_handles.pop(old_name)
        self.geometry_plot_handles[name] = temp_handle

    def update_pen(self, name, qpen):
        self.geometry_plot_handles[name].setPen(qpen)

    def clear_field(self):
        for child in self.v.allChildItems():
            if isinstance(child, pg.PColorMeshItem) or isinstance(child, PymeadPColorMeshItem):
                self.v.getViewBox().removeItem(child)
        if self.cbar is not None:
            self.w.removeItem(self.cbar)
            self.cbar = None
            self.cbar_label_attrs = None

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
        grid_stats_file = os.path.join(analysis_dir, 'mplot_grid_stats.log')
        grid_file = os.path.join(analysis_dir, f'grid.{os.path.split(analysis_dir)[-1]}')
        if not os.path.exists(field_file):
            self.disp_message_box(message=f"Field file {field_file} not found", message_mode='error')
            return
        if not os.path.exists(grid_file):
            self.disp_message_box(message=f"Grid statistics log {grid_file} not found", message_mode='error')
            return

        self.default_field_dir = analysis_dir

        # data = np.loadtxt(field_file, skiprows=2)
        field = read_field_from_mses(field_file)
        grid_stats = read_grid_stats_from_mses(grid_stats_file)
        x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
        flow_var = field[flow_var_idx[inputs['flow_variable']]]

        edgecolors = None
        antialiasing = False
        # edgecolors = {'color': 'b', 'width': 1}  # May be uncommented to see edgecolor effect
        # antialiasing = True # May be uncommented to see antialiasing effect

        pcmi_list = []

        start_idx, end_idx = 0, x_grid[0].shape[1] - 1
        for flow_section_idx in range(grid_stats["numel"] + 1):
            flow_var_section = flow_var[:, start_idx:end_idx]

            pcmi = PymeadPColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing,
                                        colorMap=pg.colormap.get('CET-R1'))
            pcmi_list.append(pcmi)
            vBox.addItem(pcmi)
            # vBox.addItem(pg.ArrowItem(pxMode=False, headLen=0.01, pos=(0.5, 0.1)))
            vBox.setAspectLocked(True, 1)
            pcmi.setZValue(0)

            pcmi.setData(x_grid[flow_section_idx], y_grid[flow_section_idx], flow_var_section)

            if flow_section_idx < grid_stats["numel"]:
                start_idx = end_idx
                end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

        for child in self.v.allChildItems():
            if hasattr(child, 'setZValue') and not isinstance(child, pg.PColorMeshItem) \
                    and not isinstance(child, PymeadPColorMeshItem):
                child.setZValue(5)

        all_levels = np.array([list(pcmi.getLevels()) for pcmi in pcmi_list])
        levels = (np.min(all_levels[:, 0]), np.max(all_levels[:, 1]))
        bar = pg.ColorBarItem(
            values=levels,
            colorMap='CET-R1',
            rounding=0.001,
            limits=levels,
            orientation='v',
            pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        )
        bar.setImageItem(pcmi_list)
        self.cbar_label_attrs = {
            'axis': 'right',
            'text': flow_var_label[inputs['flow_variable']],
            'font-size': '12pt',
        }
        if self.current_theme == "dark":
            self.cbar_label_attrs['color'] = '#dce1e6'
        elif self.current_theme == "light":
            self.cbar_label_attrs['color'] = '#000000'
        bar.setLabel(**self.cbar_label_attrs)
        self.cbar = bar
        self.w.addItem(bar)

        def on_levels_changed(cbar):
            for pcm in pcmi_list:
                pcm.setLevels(cbar.levels())

        bar.sigLevelsChanged.connect(on_levels_changed)

    def load_mea_no_dialog(self, file_name):
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.statusBar().showMessage("Loading MEA...")
        n_func_strs = count_func_strs(file_name)
        jmea = load_data(file_name)
        self.mea = MEA.generate_from_param_dict(jmea)
        self.progress_bar.setValue(10)
        self.mea.airfoil_graphs_active = True
        self.statusBar().showMessage("Adding airfoils...")
        for a in self.mea.airfoils.values():
            a.update()
        self.v.clear()
        self.param_tree_instance.t.clear()
        self.progress_bar.setValue(20)
        for idx, airfoil in enumerate(self.mea.airfoils.values()):
            self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, None, w=self.w, v=self.v, gui_obj=self)
        self.progress_bar.setValue(25)
        ProgressInfo = namedtuple("ProgressInfo", ("start", "end", "n"))
        progress_info = ProgressInfo(25, 85, n_func_strs)
        self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self, progress_info=progress_info)
        self.progress_bar.setValue(85)
        for a in self.mea.airfoils.values():
            a.airfoil_graph.param_tree = self.param_tree_instance
            a.airfoil_graph.airfoil_parameters = a.airfoil_graph.param_tree.p.param('Airfoil Parameters')
        dben = benedict.benedict(self.mea.param_dict)
        self.progress_bar.setValue(90)
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
        self.mea_start_dict = self.copy_mea_dict()
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Airfoil system load complete.", 2000)
        self.progress_bar.hide()

    def add_airfoil(self, airfoil: Airfoil):
        self.mea.te_thickness_edit_mode = self.te_thickness_edit_mode
        self.mea.add_airfoil(airfoil, len(self.mea.airfoils), self.param_tree_instance,
                             w=self.w, v=self.v, gui_obj=self)
        self.airfoil_name_list = [k for k in self.mea.airfoils.keys()]
        self.param_tree_instance.p.child("Analysis").child("Inviscid Cl Calc").setLimits([a.tag for a in self.mea.airfoils.values()])
        self.param_tree_instance.params[-1].add_airfoil(airfoil)
        for a in self.mea.airfoils.values():
            if a.airfoil_graph.airfoil_parameters is None:
                a.airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')
        airfoil.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.param_tree_instance.plot_changed,
                                                                     f"A{len(self.mea.airfoils) - 1}"))

    def disp_message_box(self, message: str, message_mode: str = 'error', rich_text: bool = False):
        disp_message_box(message, self, message_mode=message_mode, rich_text=rich_text)

    def output_area_text(self, text: str, mode: str = 'plain', mono: bool = True):
        prepend_html = f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style>" \
                       f"</head><body><p><font size='4'>&#8203;</font></p></body>"
        previous_cursor = self.text_area.textCursor()
        self.text_area.moveCursor(QTextCursor.End)
        if mode == 'plain':
            if mode == "plain" and mono:
                self.text_area.insertHtml(prepend_html)
            self.text_area.insertPlainText(text)
        elif mode == 'html':
            self.text_area.insertHtml(text)
        else:
            raise ValueError('Mode must be \'plain\' or \'html\'')
        self.text_area.setTextCursor(previous_cursor)
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def display_airfoil_statistics(self):
        airfoil_stats = AirfoilStatistics(mea=self.mea)
        dialog = AirfoilStatisticsDialog(parent=self, airfoil_stats=airfoil_stats)
        dialog.exec()

    def single_airfoil_inviscid_analysis(self):
        """Inviscid analysis not yet implemented here"""
        selected_airfoil = self.param_tree_instance.p.child('Analysis').child('Inviscid Cl Calc').value()
        body_fixed_coords = self.mea.airfoils[selected_airfoil].get_coords(body_fixed_csys=True, as_tuple=False)
        _, _, CL = single_element_inviscid(body_fixed_coords,
                                           alpha=self.mea.airfoils[selected_airfoil].alf.value * 180 / np.pi)
        print(f"{CL = }")

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

            extra_get_coords_kwargs = dict(downsample=inputs["use_downsampling"],
                                           ds_max_points=inputs["downsampling_max_pts"],
                                           ds_curve_exp=inputs["downsampling_curve_exp"]) if inputs["use_downsampling"] else {}

            if json:
                coord_dict = {}
                for a in airfoils:
                    airfoil = self.mea.airfoils[a]
                    coords = airfoil.get_coords(body_fixed_csys=False, **extra_get_coords_kwargs)
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
                        coords = airfoil.get_coords(body_fixed_csys=False, **extra_get_coords_kwargs)
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

    def show_help(self):
        HelpBrowserWindow(parent=self)

    def export_IGES(self):
        self.dialog = ExportIGESDialog(parent=self)
        if self.dialog.exec_():
            inputs = self.dialog.getInputs()
            f_ = os.path.join(inputs['dir'], inputs['file_name'])
            control_point_list = [[c.P for c in a.curve_list] for a in self.mea.airfoils.values()]
            cp_list_flattened = list(itertools.chain.from_iterable(control_point_list))
            transform_3d = Transformation3D(tx=[inputs["translation"][0]],
                                            ty=[inputs["translation"][1]],
                                            tz=[inputs["translation"][2]],
                                            sx=[inputs["scaling"][0]],
                                            sy=[inputs["scaling"][1]],
                                            sz=[inputs["scaling"][2]],
                                            rx=[inputs["rotation"][0]],
                                            ry=[inputs["rotation"][1]],
                                            rz=[inputs["rotation"][2]],
                                            rotation_units="deg",
                                            order=inputs["transformation_order"])
            cp_list_3d = []
            for cp in cp_list_flattened:
                cp = np.insert(cp, 1, 0, axis=1)
                cp_list_3d.append(cp)
            transformed_cp_list = [transform_3d.transform(P) for P in cp_list_3d]
            bez_IGES_list = [BezierIGES(P) for P in transformed_cp_list]
            iges_generator = IGESGenerator(bez_IGES_list)
            iges_generator.generate(f_)
            self.disp_message_box(f"Airfoil geometry saved to {f_}", message_mode="info")

    def single_airfoil_viscous_analysis(self):
        self.dialog = XFOILDialog(parent=self)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None

        if inputs is not None:
            xfoil_settings = {'Re': inputs['Re'],
                              'Ma': inputs['Ma'],
                              'prescribe': inputs['prescribe'],
                              'timeout': inputs['timeout'],
                              'iter': inputs['iter'],
                              'xtr': [inputs['xtr_lower'], inputs['xtr_upper']],
                              'N': inputs['N'],
                              'airfoil_analysis_dir': inputs['airfoil_analysis_dir'],
                              'airfoil_coord_file_name': inputs['airfoil_coord_file_name'],
                              'airfoil': inputs['airfoil']}
            if xfoil_settings['prescribe'] == 'Angle of Attack (deg)':
                xfoil_settings['alfa'] = inputs['alfa']
            elif xfoil_settings['prescribe'] == 'Viscous Cl':
                xfoil_settings['Cl'] = inputs['Cl']
            elif xfoil_settings['prescribe'] == 'Inviscid Cl':
                xfoil_settings['CLI'] = inputs['CLI']

            #TODO: insert downsampling step here

            coords = tuple(self.mea.deepcopy().airfoils[xfoil_settings['airfoil']].get_coords(
                body_fixed_csys=False, as_tuple=True))

            aero_data, _ = calculate_aero_data(xfoil_settings['airfoil_analysis_dir'],
                                               xfoil_settings['airfoil_coord_file_name'],
                                               coords=coords,
                                               tool="XFOIL",
                                               xfoil_settings=xfoil_settings,
                                               export_Cp=True
                                               )
            if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
                self.disp_message_box("XFOIL Analysis Failed", message_mode='error')
                self.output_area_text(
                    f"[{str(self.n_analyses).zfill(2)}] XFOIL Converged = {aero_data['converged']} | Errored out = "
                    f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}")
                self.output_area_text('\n')
            else:
                self.output_area_text(
                    f"[{str(self.n_analyses).zfill(2)}] XFOIL ({xfoil_settings['airfoil']}, "
                    f"\u03b1 = {aero_data['alf']:.3f}, Re = {xfoil_settings['Re']:.3E}, "
                    f"Ma = {xfoil_settings['Ma']:.3f}): "
                    f"Cl = {aero_data['Cl']:+7.4f} | Cd = {aero_data['Cd']:+.5f} | Cm = {aero_data['Cm']:+7.4f} "
                    f"| L/D = {aero_data['L/D']:+8.4f}".replace("-", "\u2212"))
                self.output_area_text('\n')
            bar = self.text_area.verticalScrollBar()
            sb = bar
            sb.setValue(sb.maximum())

            if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
                if self.analysis_graph is None:
                    # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow geometry docking window to be closed
                    self.analysis_graph = AnalysisGraph(background_color=self.themes[self.current_theme]["graph-background-color"])
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

        self.dialog = MultiAirfoilDialog(parent=self, settings_override=self.multi_airfoil_analysis_settings,
                                         design_tree_widget=self.design_tree_widget)
        self.dialog.show()
        self.dialog.accepted.connect(self.multi_airfoil_analysis_accepted)
        self.dialog.rejected.connect(self.multi_airfoil_analysis_rejected)

    def multi_airfoil_analysis_accepted(self):

        inputs = self.dialog.getInputs()
        self.multi_airfoil_analysis_settings = inputs

        if inputs is not None:
            mset_settings = convert_dialog_to_mset_settings(inputs['MSET'])
            mses_settings = convert_dialog_to_mses_settings(inputs['MSES'])
            mses_settings['n_airfoils'] = mset_settings['n_airfoils']
            mplot_settings = convert_dialog_to_mplot_settings(inputs['MPLOT'])
            self.multi_airfoil_analysis(mset_settings, mses_settings, mplot_settings)

    def multi_airfoil_analysis_rejected(self):
        self.multi_airfoil_analysis_settings = self.dialog.getInputs()

    def multi_airfoil_analysis(self, mset_settings: dict, mses_settings: dict,
                               mplot_settings: dict):
        # print(f"{mplot_settings = }")
        mea = self.mea.deepcopy() if mset_settings["use_downsampling"] else self.mea

        coords = tuple([mea.airfoils[k].get_coords(
            body_fixed_csys=False, as_tuple=True, downsample=mset_settings["use_downsampling"],
            ds_max_points=mset_settings["downsampling_max_pts"],
            ds_curve_exp=mset_settings["downsampling_curve_exp"]) for k in mset_settings['airfoil_order']])
        aero_data, _ = calculate_aero_data(mset_settings['airfoil_analysis_dir'],
                                           mset_settings['airfoil_coord_file_name'],
                                           coords=coords,
                                           tool='MSES',
                                           export_Cp=True,
                                           mset_settings=mset_settings,
                                           mses_settings=mses_settings,
                                           mplot_settings=mplot_settings)
        if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
            self.disp_message_box("MSES Analysis Failed", message_mode='error')
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] MSES Converged = {aero_data['converged']} | Errored out = "
                f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}")
            self.output_area_text('\n')
        else:
            # self.output_area_text('\n')
            if "L/D" not in aero_data.keys():
                aero_data["L/D"] = np.true_divide(aero_data["Cl"], aero_data["Cd"])
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] MSES  (    \u03b1 = {aero_data['alf']:.3f}, Re = {mses_settings['REYNIN']:.3E}, "
                f"Ma = {mses_settings['MACHIN']:.3f}): "
                f"Cl = {aero_data['Cl']:+7.4f} | Cd = {aero_data['Cd']:+.5f} | "
                f"Cm = {aero_data['Cm']:+7.4f} | L/D = {aero_data['L/D']:+8.4f}".replace("-", "\u2212"))
            self.output_area_text('\n')
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

        if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
            if self.analysis_graph is None:
                # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow
                # geometry docking window to be closed
                self.analysis_graph = AnalysisGraph(background_color=self.themes[self.current_theme]["graph-background-color"])
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
            # res = match_airfoil_ga(self.mea, target_airfoil, airfoil_name)
            res = match_airfoil(self.mea, target_airfoil, airfoil_name)
            msg_mode = 'error'
            if hasattr(res, 'success') and res.success or hasattr(res, 'F') and res.F is not None:
                if hasattr(res, 'x'):
                    update_params = res.x
                elif hasattr(res, 'X'):
                    update_params = res.X
                else:
                    raise AttributeError("Did not have x or X to update airfoil parameter dictionary")
                # print(f"{update_params = }")
                self.mea.update_parameters(update_params)
                msg_mode = 'info'
            self.disp_message_box(message=res.message, message_mode=msg_mode)

    def plot_airfoil_from_airfoiltools(self):
        dialog = AirfoilPlotDialog(self)
        if dialog.exec_():
            airfoil_name = dialog.getInputs()
            airfoil = extract_data_from_airfoiltools(airfoil_name)
            self.v.plot(airfoil[:, 0], airfoil[:, 1], pen=pg.mkPen(color='orange', width=1))

    def setup_optimization(self):
        self.dialog = OptimizationSetupDialog(self, settings_override=self.opt_settings,
                                              design_tree_widget=self.design_tree_widget)
        self.dialog.show()
        self.dialog.accepted.connect(self.optimization_accepted)
        self.dialog.rejected.connect(self.optimization_rejected)

    def optimization_accepted(self):
        exit_the_dialog = False
        early_return = False
        opt_settings_list = None
        param_dict_list = None
        mea_list = None
        files = None
        while not exit_the_dialog and not early_return:
            self.opt_settings = self.dialog.getInputs()

            loop_through_settings = False

            if self.opt_settings['General Settings']['batch_mode_active']:

                loop_through_settings = True

                files = self.opt_settings['General Settings']['batch_mode_files']

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

                param_dict = deepcopy(convert_opt_settings_to_param_dict(opt_settings))

                # First check to make sure MSET, MSES, and MPLOT can be found on system path and marked as executable:
                if param_dict["tool"] == "XFOIL" and shutil.which('xfoil') is None:
                    self.disp_message_box('XFOIL executable \'xfoil\' not found on system path')
                    return
                if param_dict["tool"] == "MSES" and shutil.which('mset') is None:
                    self.disp_message_box('MSES suite executable \'mset\' not found on system path')
                    return
                if param_dict["tool"] == "MSES" and shutil.which('mses') is None:
                    self.disp_message_box('MSES suite executable \'mses\' not found on system path')
                    return
                if param_dict["tool"] == "MSES" and shutil.which('mplot') is None:
                    self.disp_message_box('MPLOT suite executable \'mplot\' not found on system path')
                    return

                # print(f"{opt_settings['General Settings']['use_current_mea'] = }")
                if opt_settings['General Settings']['use_current_mea']:
                    mea_dict = self.mea.copy_as_param_dict(deactivate_airfoil_graphs=True)
                else:
                    mea_file = opt_settings['General Settings']['mea_file']
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

                # CONSTRAINTS
                for airfoil_name, constraint_set in param_dict['constraints'].items():

                    # Thickness distribution check parameters
                    if constraint_set['check_thickness_at_points']:
                        thickness_file = constraint_set['thickness_at_points']
                        try:
                            data = np.loadtxt(thickness_file)
                            param_dict['constraints'][airfoil_name]['thickness_at_points'] = data.tolist()
                        except FileNotFoundError:
                            message = f'Thickness file {thickness_file} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        constraint_set['thickness_at_points'] = None

                    # Internal geometry check parameters
                    if constraint_set['use_internal_geometry']:
                        internal_geometry_file = constraint_set['internal_geometry']
                        try:
                            data = np.loadtxt(internal_geometry_file)
                            constraint_set['internal_geometry'] = data.tolist()
                        except FileNotFoundError:
                            message = f'Internal geometry file {internal_geometry_file} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        constraint_set['internal_geometry'] = None

                    # External geometry check parameters
                    if constraint_set['use_external_geometry']:
                        external_geometry_file = constraint_set['external_geometry']
                        try:
                            data = np.loadtxt(external_geometry_file)
                            constraint_set['external_geometry'] = data.tolist()
                        except FileNotFoundError:
                            message = f'External geometry file {external_geometry_file} not found'
                            self.disp_message_box(message=message, message_mode='error')
                            raise FileNotFoundError(message)
                    else:
                        constraint_set['external_geometry'] = None

                # MULTI-POINT OPTIMIZATION
                multi_point_stencil = None
                if opt_settings['Multi-Point Optimization']['multi_point_active']:
                    try:
                        multi_point_data = np.loadtxt(param_dict['multi_point_stencil'], delimiter=',')
                        multi_point_stencil = read_stencil_from_array(multi_point_data, tool=param_dict["tool"])
                    except FileNotFoundError:
                        message = f'Multi-point stencil file {param_dict["multi_point_stencil"]} not found'
                        self.disp_message_box(message=message, message_mode='error')
                        raise FileNotFoundError(message)
                if param_dict['tool'] == 'MSES':
                    param_dict['mses_settings']['multi_point_stencil'] = multi_point_stencil
                elif param_dict['tool'] == 'XFOIL':
                    param_dict['xfoil_settings']['multi_point_stencil'] = multi_point_stencil
                else:
                    raise ValueError(f"Currently only MSES and XFOIL are supported as analysis tools for "
                                     f"aerodynamic shape optimization. Tool selected was {param_dict['tool']}")

                # Warm start parameters
                if opt_settings['General Settings']['warm_start_active']:
                    opt_dir = opt_settings['General Settings']['warm_start_dir']
                else:
                    opt_dir = make_ga_opt_dir(opt_settings['Genetic Algorithm']['root_dir'],
                                              opt_settings['Genetic Algorithm']['opt_dir_name'])

                param_dict['opt_dir'] = opt_dir
                self.current_opt_folder = opt_dir.replace(os.sep, "/")

                name_base = 'ga_airfoil'
                name = [f"{name_base}_{i}" for i in range(opt_settings['Genetic Algorithm']['n_offspring'])]
                param_dict['name'] = name

                for airfoil in mea.airfoils.values():
                    airfoil.airfoil_graphs_active = False
                mea.airfoil_graphs_active = False
                base_folder = os.path.join(opt_settings['Genetic Algorithm']['root_dir'],
                                           opt_settings['Genetic Algorithm']['temp_analysis_dir_name'])
                param_dict['base_folder'] = base_folder
                if not os.path.exists(base_folder):
                    os.mkdir(base_folder)

                if opt_settings['General Settings']['warm_start_active']:
                    param_dict['warm_start_generation'] = calculate_warm_start_index(
                        opt_settings['General Settings']['warm_start_generation'], opt_dir)
                    if param_dict['warm_start_generation'] == 0:
                        opt_settings['General Settings']['warm_start_active'] = False
                param_dict_save = deepcopy(param_dict)
                if not opt_settings['General Settings']['warm_start_active']:
                    save_data(param_dict_save, os.path.join(opt_dir, 'param_dict.json'))
                else:
                    save_data(param_dict_save, os.path.join(
                        opt_dir, f'param_dict_{param_dict["warm_start_generation"]}.json'))

                if not loop_through_settings:
                    opt_settings_list = [opt_settings]
                param_dict_list.append(param_dict)
                mea_list.append(mea)
                exit_the_dialog = True

        if early_return:
            self.setup_optimization()

        if not early_return:
            for (opt_settings, param_dict, mea) in zip(opt_settings_list, param_dict_list, mea_list):
                # The next line is just to make sure any calls to the GUI are performed before the optimization
                self.dialog.overrideInputs(new_inputs=opt_settings)

                # Need to regenerate the objectives and constraints here since they contain references to
                # (non-serializable) modules which must be passed through a multiprocessing.Pipe
                new_obj_list = [obj.func_str for obj in self.objectives]
                new_constr_list = [constr.func_str for constr in self.constraints]

                # Run the shape optimization in a worker thread
                self.run_shape_optimization(param_dict, opt_settings,
                                            mea.copy_as_param_dict(deactivate_airfoil_graphs=True),
                                            new_obj_list, new_constr_list,
                                            )

    def optimization_rejected(self):
        self.opt_settings = self.dialog.getInputs()
        return

    @pyqtSlot(str, object)
    def progress_update(self, status: str, data: object):
        bcolor = self.themes[self.current_theme]["graph-background-color"]
        if status == "text" and isinstance(data, str):
            self.output_area_text(data)
        elif status == "message" and isinstance(data, str):
            self.message_callback_fn(data)
        elif status == "opt_progress" and isinstance(data, dict):
            callback = TextCallback(parent=self, text_list=data["text"], completed=data["completed"],
                                    warm_start_gen=data["warm_start_gen"])
            callback.exec_callback()
        elif status == "airfoil_coords" and isinstance(data, list):
            callback = PlotAirfoilCallback(parent=self, coords=data,
                                           background_color=bcolor)
            callback.exec_callback()
        elif status == "parallel_coords" and isinstance(data, tuple):
            callback = ParallelCoordsCallback(parent=self, norm_val_list=data[0], param_name_list=data[1],
                                              background_color=bcolor)
            callback.exec_callback()
        elif status == "cp_xfoil":
            callback = CpPlotCallbackXFOIL(parent=self, Cp=data, background_color=bcolor)
            callback.exec_callback()
        elif status == "cp_mses":
            callback = CpPlotCallbackMSES(parent=self, Cp=data, background_color=bcolor)
            callback.exec_callback()
        elif status == "drag_xfoil" and isinstance(data, tuple):
            callback = DragPlotCallbackXFOIL(parent=self, Cd=data[0], Cdp=data[1], Cdf=data[2], background_color=bcolor)
            callback.exec_callback()
        elif status == "drag_mses" and isinstance(data, tuple):
            callback = DragPlotCallbackMSES(parent=self, Cd=data[0], Cdp=data[1], Cdf=data[2], Cdv=data[3], Cdw=data[4],
                                            background_color=bcolor)
            callback.exec_callback()

    def clear_opt_plots(self):
        def clear_handles(h_list: list):
            for h in h_list:
                if isinstance(h, list):
                    for h_ in h:
                        h_.clear()
                else:
                    h.clear()
            h_list.clear()

        for handle_list in [self.opt_airfoil_plot_handles, self.parallel_coords_plot_handles,
                            self.Cp_graph_plot_handles]:
            clear_handles(handle_list)

        if self.drag_graph is not None:
            for Cd_val in ["Cd", "Cdp", "Cdv", "Cdw", "Cdf"]:
                if hasattr(self.drag_graph, f"pg_plot_handle_{Cd_val}"):
                    handle = getattr(self.drag_graph, f"pg_plot_handle_{Cd_val}")
                    handle.clear()

    def run_shape_optimization(self, param_dict: dict, opt_settings: dict, mea: dict, objectives, constraints):

        self.clear_opt_plots()

        def run_cpu_bound_process():
            shape_opt_process = CPUBoundProcess(
                shape_optimization_static,
                args=(param_dict, opt_settings, mea, objectives, constraints)
            )
            shape_opt_process.progress_emitter.signals.progress.connect(self.progress_update)
            shape_opt_process.progress_emitter.signals.finished.connect(self.shape_opt_finished_callback_fn)
            shape_opt_process.start()
            self.shape_opt_process = shape_opt_process

        # Start running the CPU-bound process from a worker thread (separate from the main GUI thread)
        thread = Thread(target=run_cpu_bound_process)
        self.opt_thread = thread
        self.opt_thread.start()

        # TODO: follow this same code architecture for XFOIL and MSES one-off analysis

    def stop_optimization(self):

        # Close the Pipe, which triggers either an immediate termination of the worker process or a termination of the
        # Pool after the next CFD evaluation completes, depending on the progress of the optimization.
        self.shape_opt_process.parent_conn.close()
        self.shape_opt_process.child_conn.close()

        if self.opt_thread is not None:
            self.opt_thread.join()

        print("Optimization terminated.")

    @staticmethod
    def generate_output_folder_link_text(folder: str):
        return f"<a href='{folder}' style='font-family:DejaVu Sans Mono; " \
               f"color: #1FBBCC; font-size: 14px;'>Open output folder</a>\n"

    def set_pool(self, pool_obj: object):
        print(f"Setting pool! {pool_obj = }")
        self.pool = pool_obj

    @staticmethod
    def shape_opt_progress_callback_fn(progress_object: object):
        if isinstance(progress_object, OptCallback):
            progress_object.exec_callback()

    def shape_opt_finished_callback_fn(self, success: bool):
        self.forces_dict = {}
        self.pool = None
        self.status_bar.showMessage("Optimization Complete!", 3000)
        first_word = "Completed" if success else "Terminated"
        self.output_area_text(f"{first_word} aerodynamic shape optimization. ")
        self.output_area_text(self.generate_output_folder_link_text(self.current_opt_folder), mode="html")
        self.output_area_text("\n\n")
        self.current_opt_folder = None
        self.status_bar.showMessage("")
        # self.finished_optimization = True

    def shape_opt_result_callback_fn(self, result_object: object):
        pass

    def message_callback_fn(self, message: str):
        self.status_bar.showMessage(message)

    def text_area_callback_fn(self, message: str):
        self.output_area_text(message)

    def shape_opt_error_callback_fn(self, error_tuple: tuple):
        self.output_area_text(f"Error. Error = {error_tuple}\n")

    def write_force_dict_to_file(self, file_name: str):
        forces_temp = deepcopy(self.forces_dict)
        if "Cp" in forces_temp.keys():
            for el in forces_temp["Cp"]:
                if isinstance(el, list):
                    for e in el:
                        for k, v in e.items():
                            if isinstance(v, np.ndarray):
                                e[k] = v.tolist()
                else:
                    for k, v in el.items():
                        if isinstance(v, np.ndarray):
                            el[k] = v.tolist()
        save_data(forces_temp, file_name)

    def read_force_dict_from_file(self, file_name: str):
        self.forces_dict = load_data(file_name)

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


# Adjustments for variable monitor resolution (from https://stackoverflow.com/a/47723454)
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def main():
    app = QApplication(sys.argv)
    app.processEvents()
    app.setStyle('Fusion')
    if len(sys.argv) > 1:
        gui = GUI(path=sys.argv[1])
    else:
        gui = GUI()

    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # First, we must add freeze support for multiprocessing.Pool to work properly in Windows in the version of the GUI
    # assembled by PyInstaller. This next statement affects only Windows; it has no impact on *nix OS since Pool
    # already works fine there.
    mp.freeze_support()

    # Generate the graphical user interface
    main()
