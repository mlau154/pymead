import multiprocessing as mp
import os
import shutil
import sys
import typing
from collections import namedtuple
from copy import deepcopy
from functools import partial
from threading import Thread

import numpy as np
import pyqtgraph as pg
import requests
from PyQt5.QtCore import QEvent, QObject, Qt, QThreadPool, QRect
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QFont, QFontDatabase, QPainter, QCloseEvent, QTextCursor
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QMainWindow, QApplication, \
    QWidget, QMenu, QStatusBar, QAction, QGraphicsScene, QGridLayout, QDockWidget, QSizeGrip
from cmcrameri import cm
from pymoo.factory import get_decomposition
from pyqtgraph.exporters import CSVExporter, SVGExporter

from pymead import ICON_DIR, GUI_SETTINGS_DIR, GUI_THEMES_DIR, GUI_DEFAULT_AIRFOIL_DIR, q_settings
from pymead import RESOURCE_DIR
from pymead.analysis.calc_aero_data import SVG_PLOTS, SVG_SETTINGS_TR
from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.analysis.read_aero_data import flow_var_idx
from pymead.analysis.read_aero_data import read_grid_stats_from_mses, read_field_from_mses, \
    read_streamline_grid_from_mses
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.mea import MEA
from pymead.gui.airfoil_canvas import AirfoilCanvas
from pymead.gui.airfoil_statistics import AirfoilStatisticsDialog, AirfoilStatistics
from pymead.gui.analysis_graph import AnalysisGraph
from pymead.gui.concurrency import CPUBoundProcess
from pymead.gui.custom_graphics_view import CustomGraphicsView
from pymead.gui.dockable_tab_widget import PymeadDockWidget
from pymead.gui.help_browser import HelpBrowserWindow
from pymead.gui.input_dialog import LoadDialog, SaveAsDialog, OptimizationSetupDialog, \
    MultiAirfoilDialog, ColorInputDialog, ExportCoordinatesDialog, ExportControlPointsDialog, AirfoilPlotDialog, \
    AirfoilMatchingDialog, MSESFieldPlotDialog, ExportIGESDialog, XFOILDialog, NewGeoColDialog, EditBoundsDialog, \
    ExitDialog, ScreenshotDialog, LoadAirfoilAlgFile, ExitOptimizationDialog
from pymead.gui.input_dialog import convert_dialog_to_mset_settings, convert_dialog_to_mses_settings, \
    convert_dialog_to_mplot_settings
from pymead.gui.main_icon_toolbar import MainIconToolbar
from pymead.gui.message_box import disp_message_box
from pymead.gui.parameter_tree import ParameterTree
from pymead.gui.permanent_widget import PermanentWidget
from pymead.gui.pymeadPColorMeshItem import PymeadPColorMeshItem
from pymead.gui.show_hide import ShowHideDialog
from pymead.gui.side_grip import SideGrip
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.title_bar import TitleBar
from pymead.optimization.opt_callback import PlotAirfoilCallback, ParallelCoordsCallback, OptCallback, \
    DragPlotCallbackXFOIL, CpPlotCallbackXFOIL, DragPlotCallbackMSES, CpPlotCallbackMSES, TextCallback
from pymead.optimization.opt_setup import calculate_warm_start_index
from pymead.optimization.opt_setup import read_stencil_from_array, convert_opt_settings_to_param_dict
from pymead.optimization.shape_optimization import shape_optimization as shape_optimization_static
from pymead.post.mses_field import flow_var_label
from pymead.utils.airfoil_matching import match_airfoil
from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.utils.misc import count_func_strs, get_setting
from pymead.utils.misc import make_ga_opt_dir
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.version_check import using_latest


class GUI(QMainWindow):
    _gripSize = 5

    def __init__(self, path=None, parent=None):
        # super().__init__(flags=Qt.FramelessWindowHint)
        super().__init__(parent=parent)
        self.showHideState = None
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.windowMaximized = False
        # print(f"Running GUI with {os.getpid() = }")
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
        self.crameri_cmap = None
        self.default_field_dir = None
        self.objectives = []
        self.constraints = []
        self.airfoil_name_list = []
        self.last_analysis_dir = None
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
        self.pens = [
            ('#d4251c', Qt.SolidLine),
            ('darkorange', Qt.SolidLine),
            ('gold', Qt.SolidLine),
            ('limegreen', Qt.SolidLine),
            ('cyan', Qt.SolidLine),
            ('mediumpurple', Qt.SolidLine),
            ('deeppink', Qt.SolidLine),
            ('#d4251c', Qt.DashLine),
            ('darkorange', Qt.DashLine),
            ('gold', Qt.DashLine),
            ('limegreen', Qt.DashLine),
            ('cyan', Qt.DashLine),
            ('mediumpurple', Qt.DashLine),
            ('deeppink', Qt.DashLine)
        ]

        self.current_save_name = None

        # mandatory for cursor updates
        self.setMouseTracking(True)

        self.title_bar = TitleBar(self)
        self.title_bar.sigMessage.connect(self.disp_message_box)

        self.sideGrips = [
            SideGrip(self, Qt.LeftEdge),
            SideGrip(self, Qt.TopEdge),
            SideGrip(self, Qt.RightEdge),
            SideGrip(self, Qt.BottomEdge),
        ]
        # corner grips should be "on top" of everything, otherwise the side grips
        # will take precedence on mouse events, so we are adding them *after*;
        # alternatively, widget.raise_() can be used
        self.cornerGrips = [QSizeGrip(self) for i in range(4)]

        # Dock widget items
        self.dock_widgets = []
        self.dock_widget_names = []
        self.first_dock_widget = None
        self.current_dock_widget = None
        self.tabifiedDockWidgetActivated.connect(self.activated)
        self.setDockNestingEnabled(True)

        self.status_bar = QStatusBar()
        self.status_bar.messageChanged.connect(self.statusChanged)
        self.setStatusBar(self.status_bar)

        self.text_area = ConsoleTextArea(self)

        self.geo_col = GeometryCollection(gui_obj=self)

        self.airfoil_canvas = AirfoilCanvas(self, geo_col=self.geo_col, gui_obj=self)
        self.airfoil_canvas.sigStatusBarUpdate.connect(self.setStatusBarText)

        self.parameter_tree = ParameterTree(geo_col=self.geo_col, parent=self, gui_obj=self)

        self.geo_col.add_param(value=0.5, name="test_param")

        self.add_new_tab_widget(self.parameter_tree, "Tree")
        self.add_new_tab_widget(self.airfoil_canvas, "Geometry")
        self.add_new_tab_widget(self.text_area, "Console")

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

        # self.auto_range_geometry()
        self.statusBar().clearMessage()
        self.permanent_widget = PermanentWidget(self)
        self.statusBar().addPermanentWidget(self.permanent_widget)

        self.parameter_tree.setMaximumWidth(300)
        self.resize(1000, self.title_bar.height() + 600)
        self.parameter_tree.setMaximumWidth(800)

        # Load the airfoil system from the system argument variable if necessary
        # self.mea_start_dict = None
        # if self.path is not None:
        #     self.load_mea_no_dialog(self.path)
        # else:
        #     self.mea_start_dict = self.copy_mea_dict()

        # Check if we are using the most recent release of pymead (notify if not)
        self.check_for_new_version()

    def check_for_new_version(self):
        try:
            using_latest_res, latest_ver, current_ver = using_latest()
        except requests.ConnectionError:
            self.disp_message_box("Could not connect to the internet to check for updates", message_mode="info")
            return

        if not using_latest_res:
            self.disp_message_box(f"A newer version of pymead ({latest_ver}) is available for download at "
                                  f"<a href='https://github.com/mlau154/pymead/releases' style='color:#45C5E6;'>"
                                  f"https://github.com/mlau154/pymead/releases</a>", message_mode="info",
                                  rich_text=True)

    def add_new_tab_widget(self, widget, name):
        if not (name in self.dock_widget_names):
            dw = PymeadDockWidget(name, self)
            dw.setAllowedAreas(Qt.AllDockWidgetAreas)
            dw.setWidget(widget)
            dw.setFloating(False)
            if name in ["Tree", "Geometry", "Console"]:
                dw.setFeatures(dw.features() & ~QDockWidget.DockWidgetClosable)
            dw.tab_closed.connect(self.on_tab_closed)
            self.dock_widgets.append(dw)
            self.dock_widget_names.append(name)
            if len(self.dock_widgets) == 2:
                self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widgets[-2])  # Left
                self.addDockWidget(Qt.RightDockWidgetArea, dw)  # Right
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.Horizontal)
            elif len(self.dock_widgets) == 3:
                self.addDockWidget(Qt.RightDockWidgetArea, dw)
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.Vertical)
            elif len(self.dock_widgets) == 4:
                self.addDockWidget(Qt.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-3], self.dock_widgets[-1])
            elif len(self.dock_widgets) > 4:
                self.addDockWidget(Qt.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])

    def on_tab_closed(self, name: str, event: QCloseEvent):
        if name == "Analysis":
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
        idx = self.dock_widget_names.index(name)
        self.dock_widget_names.pop(idx)
        self.dock_widgets.pop(idx)

    def activated(self, dw: QDockWidget):
        self.current_dock_widget = dw

    def closeEvent(self, a0) -> None:
        """
        Close Event handling for the GUI, allowing changes to be saved before exiting the program.

        Parameters
        ==========
        a0: QCloseEvent
            Qt CloseEvent object
        """
        if self.shape_opt_process is not None:
            dialog = ExitOptimizationDialog(self, theme=self.themes[self.current_theme])
            if dialog.exec_():
                self.stop_optimization()
            else:
                a0.ignore()
                return

        if self.mea_start_dict != self.copy_mea_dict():  # Only run this code if changes have been made
            save_dialog = NewGeoColDialog(parent=self)
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

    def changeEvent(self, a0):
        if a0.type() == QEvent.Type.WindowStateChange:
            self.title_bar.windowStateChanged(self.windowState())
            self.windowMaximized = self.windowState() == Qt.WindowMaximized

            # Disable the resize grips when the window is maximized and enable them otherwise
            if self.windowMaximized:
                for grip in self.sideGrips:
                    grip.setDisabled(True)
            else:
                for grip in self.sideGrips:
                    grip.setEnabled(True)

            super().changeEvent(a0)
            a0.accept()

    #
    # def resizeEvent(self, a0):
    #     self.title_bar.resize(self.width(), self.title_bar.height())
    #     super().resizeEvent(a0)
    #     a0.accept()

    @property
    def gripSize(self):
        return self._gripSize

    def setGripSize(self, size):
        if size == self._gripSize:
            return
        self._gripSize = max(2, size)
        self.updateGrips()

    def updateGrips(self):
        self.setContentsMargins(self.gripSize, self.gripSize + self.title_bar.height(), self.gripSize, self.gripSize)

        outRect = self.rect()
        # an "inner" rect used for reference to set the geometries of size grips
        inRect = outRect.adjusted(self.gripSize, self.gripSize, -self.gripSize, -self.gripSize)

        # top left
        self.cornerGrips[0].setGeometry(QRect(outRect.topLeft(), inRect.topLeft()))
        # top right
        self.cornerGrips[1].setGeometry(QRect(outRect.topRight(), inRect.topRight()).normalized())
        # bottom right
        self.cornerGrips[2].setGeometry(QRect(inRect.bottomRight(), outRect.bottomRight()))
        # bottom left
        self.cornerGrips[3].setGeometry(QRect(outRect.bottomLeft(), inRect.bottomLeft()).normalized())

        # left edge
        self.sideGrips[0].setGeometry(0, inRect.top(), self.gripSize, inRect.height())
        # top edge
        self.sideGrips[1].setGeometry(inRect.left(), 0, inRect.width(), self.gripSize)
        # right edge
        self.sideGrips[2].setGeometry(inRect.left() + inRect.width(), inRect.top(), self.gripSize, inRect.height())
        # bottom edge
        self.sideGrips[3].setGeometry(self.gripSize, inRect.top() + inRect.height(), inRect.width(), self.gripSize)

    def resizeEvent(self, event):
        self.title_bar.resize(self.width(), self.title_bar.height())
        super().resizeEvent(event)
        self.updateGrips()

    @pyqtSlot(str, int)
    def setStatusBarText(self, message: str, msecs: int):
        self.statusBar().showMessage(message, msecs)

    def set_theme(self, theme_name: str):
        self.current_theme = theme_name
        theme = self.themes[theme_name]
        self.setStyleSheet(f"""background-color: {theme['background-color']};
                           color: {theme['main-color']}; font: 10pt DejaVu Sans;
                           """
                           )
        self.text_area.setStyleSheet(f"""background-color: {theme['console-background-color']};""")
        self.title_bar.setStyleSheet(f"QLabel {{ color: {theme['main-color']}; }}")
        self.title_bar.title.setStyleSheet(
            f"""background-color: qlineargradient(x1: 0.0, y1: 0.5, x2: 1.0, y2: 0.5, 
            stop: 0 {theme['background-color']}, 
            stop: 0.5 {theme['title-gradient-color']}, 
            stop: 1 {theme['background-color']})""")

        # Set title bar buttons
        self.title_bar.normalButton.setIcon(QIcon(os.path.join(ICON_DIR, f"normal-{theme_name}-mode.svg")))
        self.title_bar.minimizeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"minimize-{theme_name}-mode.svg")))
        self.title_bar.maximizeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"maximize-{theme_name}-mode.svg")))
        self.title_bar.closeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"close-{theme_name}-mode.svg")))

        # Set menu bar sheet
        self.menuBar().setStyleSheet(f"""
                           QMenuBar {{ background-color: {theme['menu-background-color']}; font-family: "DejaVu Sans" 
                            }}
                            QMenuBar::item:selected {{ background: {theme['menu-item-selected-color']} }}
                           QMenu {{ background-color: {theme['menu-background-color']}; 
                           color: {theme['menu-main-color']};}} 
                           QMenu::item:selected {{ background-color: {theme['menu-item-selected-color']}; }}
                    """)
        self.parameter_tree.setAutoFillBackground(True)
        self.parameter_tree.setStyleSheet(
            f"""QTreeWidget::item {{ background-color: {theme['tree-background-color']}; }} 
                QTreeWidget::item:selected {{ background-color: {theme['menu-item-selected-color']}; color: {theme['main-color']} }}
                QTreeWidget::item:hover {{ color: #edb126 }}
                QTreeWidget {{ background-color: {theme['tree-background-color']} }}
                QTreeWidget::branch {{ background: {theme['tree-background-color']} }}
                QTreeWidget::branch::closed::has-children {{
                    image: url(../icons/closed-arrow-{self.current_theme}.png);
                }}            
        
                QTreeWidget::branch::open::has-children {{
                    image: url(../icons/opened-arrow-{self.current_theme}.png);
                }}
             """)
        self.parameter_tree.setForegroundColorAllItems(theme['main-color'])
        self.airfoil_canvas.setAxisLabels(theme)

        for dock_widget in self.dock_widgets:
            if hasattr(dock_widget.widget(), 'setBackground'):
                dock_widget.widget().setBackground(theme["dock-widget-background-color"])
        if self.cbar is not None and self.cbar_label_attrs is not None:
            self.cbar_label_attrs["color"] = theme["cbar-color"]
            self.cbar.setLabel(**self.cbar_label_attrs)
            self.cbar.setColorMap(self.crameri_cmap[self.current_theme])
            tick_font = QFont(get_setting("cbar-tick-font-family"), get_setting("cbar-tick-point-size"))
            self.cbar.axis.setStyle(tickFont=tick_font)
            self.cbar.axis.setTextPen(pg.mkPen(color=theme["cbar-color"]))
        # if self.analysis_graph is not None:
        #     self.analysis_graph.set_background(theme["graph-background-color"])
        # if self.param_tree_instance is not None:
        #     self.param_tree_instance.set_theme(theme)

        for cnstr in self.geo_col.container()["geocon"].values():
            cnstr.canvas_item.setStyle(theme)

    def set_color_bar_style(self, new_values: dict = None):
        if self.cbar is None:
            return

        new_values = {} if new_values is None else new_values
        self.cbar.setLabel(axis="right", )
        self.cbar.getAxis("right").setWidth(25 + 1)

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(ICON_DIR, "pymead-logo.png")
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        menu_data = load_data(os.path.join(GUI_SETTINGS_DIR, "menu.json"))

        def recursively_add_menus(menu: dict, menu_bar: QObject):
            for key, val in menu.items():
                if isinstance(val, dict):
                    menu = QMenu(key, parent=menu_bar)
                    menu_bar.addMenu(menu)
                    recursively_add_menus(val, menu_bar.children()[-1])
                else:
                    action = QAction(key, parent=menu_bar)
                    action_parent = action.parent()
                    if isinstance(action_parent, QMenu):
                        action_parent.addAction(action)
                    else:
                        raise ValueError('Attempted to add QAction to an object not of type QMenu')
                    if isinstance(val, list):
                        if len(val) > 3 and val[3] is not None:
                            action.triggered.connect(partial(getattr(self, val[0]), *val[3]))
                        else:
                            action.triggered.connect(getattr(self, val[0]))
                        if val[1] is not None:
                            action.setShortcut(val[1])
                        if len(val) > 2 and val[2] is not None:
                            action.setCheckable(True)
                            action.setChecked(val[2])
                    else:
                        action.triggered.connect(getattr(self, val))

        recursively_add_menus(menu_data, self.menu_bar)

    def take_screenshot(self):

        # if hasattr(self.dockable_tab_window, "current_dock_widget"):
        if self.current_dock_widget is not None:
            analysis_id = self.current_dock_widget.winId()
        else:
            analysis_id = self.dock_widgets[-1].winId()

        id_dict = {
            "Full Window": self.winId(),
            # "Parameter Tree": self.param_tree_instance.t.winId(),
            "Geometry": self.dock_widgets[0].winId(),
            "Analysis": analysis_id,
            "Console": self.text_area.winId()
        }

        dialog = ScreenshotDialog(self, theme=self.themes[self.current_theme])
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

    def showHidePymeadObjs(self, sub_container: str, show: bool):
        if show:
            self.airfoil_canvas.showPymeadObjs(sub_container)
        else:
            self.airfoil_canvas.hidePymeadObjs(sub_container)

    def openShowHidePymeadObjDialog(self):
        if self.showHideState is None:
            self.showAllPymeadObjs()
        dialog = ShowHideDialog(self, state=self.showHideState, theme=self.themes[self.current_theme])
        dialog.exec_()

    def showAllPymeadObjs(self):
        self.showHideState = self.airfoil_canvas.showAllPymeadObjs()

    def hideAllPymeadObjs(self):
        self.showHideState = self.airfoil_canvas.hideAllPymeadObjs()

    def save_as_geo_col(self):
        dialog = SaveAsDialog(self)
        if dialog.exec_():
            self.current_save_name = dialog.selectedFiles()[0]
            if self.current_save_name[-5:] != '.jmea':
                self.current_save_name += '.jmea'
            self.save_geo_col()
            self.setWindowTitle(f"pymead - {os.path.split(self.current_save_name)[-1]}")
            self.disp_message_box(f"Multi-element airfoil saved as {self.current_save_name}", message_mode='info')
            return True
        else:
            if self.save_attempts > 0:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')
            return False

    def save_geo_col(self):
        # if self.mea.file_name is None:
        if self.current_save_name is None:
            if self.save_attempts < 1:
                self.save_attempts += 1
                return self.save_as_geo_col()
            else:
                self.save_attempts = 0
                self.disp_message_box('No file name specified. File not saved.', message_mode='warn')
                return False
        else:
            save_data(self.geo_col.get_dict_rep(), self.current_save_name)
            self.setWindowTitle(f"pymead - {os.path.split(self.current_save_name)[-1]}")
            self.save_attempts = 0
            return True

    def deepcopy_geo_col(self):
        return deepcopy(self.geo_col)

    def copy_mea_dict(self, deactivate_airfoil_graphs: bool = False):
        return self.mea.copy_as_param_dict(deactivate_airfoil_graphs=deactivate_airfoil_graphs)

    def load_geo_col(self):

        # if self.mea_start_dict is not None:
        #     if self.mea_start_dict != self.copy_mea_dict():
        #         save_dialog = NewMEADialog(parent=self, message="Airfoil has changes. Save?")
        #         exit_dialog = ExitDialog(parent=self, window_title="Load anyway?",
        #                                  message="Airfoil not saved.\nAre you sure you want to load a new one?")
        #         while True:
        #             if save_dialog.exec_():  # If "Yes" to "Save Changes,"
        #                 if save_dialog.save_successful:  # If the changes were saved successfully, close the program.
        #                     break
        #                 else:
        #                     if exit_dialog.exec_():  # Otherwise, If "Yes" to "Exit the Program Anyway," close the program.
        #                         break
        #                 if save_dialog.reject_changes:  # If "No" to "Save Changes," do not load an MEA.
        #                     return
        #             else:  # If "Cancel" to "Save Changes," do not load an MEA
        #                 return

        # TODO: reimplement this logic

        dialog = LoadDialog(self, settings_var="jmea_default_open_location")

        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            file_name_parent_dir = os.path.dirname(file_name)
            q_settings.setValue(dialog.settings_var, file_name_parent_dir)
        else:
            file_name = None
        if file_name is not None:
            self.load_geo_col_no_dialog(file_name)
            self.setWindowTitle(f"pymead - {os.path.split(file_name)[-1]}")

    def new_geo_col(self):
        dialog = NewGeoColDialog(self)
        if dialog.exec_():
            self.load_geo_col_no_dialog(os.path.join(GUI_DEFAULT_AIRFOIL_DIR, "default_airfoil.jmea"))
            self.mea.file_name = None
            self.setWindowTitle(f"pymead")

    def edit_bounds(self):
        bv_dialog = EditBoundsDialog(geo_col=self.geo_col, theme=self.themes[self.current_theme], parent=self)
        bv_dialog.exec_()

    def auto_range_geometry(self):
        x_data_range, y_data_range = self.airfoil_canvas.getPointRange()
        self.airfoil_canvas.plot.getViewBox().setRange(xRange=x_data_range, yRange=y_data_range)

    def update_airfoil_parameters_from_vector(self, param_vec: np.ndarray):
        for airfoil in self.mea.airfoils.values():
            airfoil.airfoil_graph.airfoil_parameters = self.param_tree_instance.p.param('Airfoil Parameters')

        N_dv = self.mea.count_design_variables()
        if N_dv != len(param_vec):
            self.disp_message_box("Number of parameters in parameter vector does not match number of design variables "
                                  "found in the airfoil system. Check that the currently displayed airfoil system"
                                  " matches the one used to generate the parameter vector.")
            return

        try:
            self.mea.update_parameters(param_vec)
        except:
            self.disp_message_box("Could not load parameters into airfoil. Check that the current airfoil system"
                                  " displayed matches the one used in the optimization.")
            return

        self.param_tree_instance.plot_change_recursive(
            self.param_tree_instance.p.param('Airfoil Parameters').child('Custom').children())

    def import_design_variable_values(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, settings_var="parameter_list_default_open_location", file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            q_settings.setValue(dialog.settings_var, os.path.dirname(file_name))
            param_vec = np.loadtxt(file_name).tolist()
            if isinstance(param_vec, float):
                param_vec = [param_vec]
            self.geo_col.assign_design_variable_values(param_vec, bounds_normalized=True)

    def import_algorithm_pkl_file(self):
        dialog = LoadAirfoilAlgFile(self)
        if dialog.exec_():
            inputs = dialog.getInputs()
            dialog.load_airfoil_alg_file_widget.assignQSettings(inputs)

            try:
                alg = load_data(inputs["pkl_file"])
            except:
                self.disp_message_box("Could not load .pkl file. Check that the file selected is of the form"
                                      " algorithm_gen_XX.pkl.")
                return

            try:
                X = alg.opt.get("X")
                F = alg.opt.get("F")
            except AttributeError:
                self.disp_message_box("Algorithm file not recognized. Check that the file selected is of the form"
                                      " algorithm_gen_XX.pkl.")
                return

            if len(F) == 0:
                self.disp_message_box("Empty optimization result")
                return

            if alg.problem.n_obj == 1:  # If single-objective:
                x = X[0, :]
            else:  # If multi-objective
                if inputs["pkl_use_index"]:
                    x = X[inputs["pkl_index"], :]
                elif inputs["pkl_use_weights"]:
                    decomp = get_decomposition("asf")

                    if len(inputs["pkl_weights"]) == 0:
                        self.disp_message_box("The requested weights do not sum to 1.0.")
                        return

                    if len(inputs["pkl_weights"]) != F.shape[1]:
                        self.disp_message_box(f"Length of the requested weight list ({len(inputs['pkl_weights'])}) does"
                                              f" not match the number of objective functions ({F.shape[1]})")
                        return

                    IDX = decomp.do(F, inputs["pkl_weights"]).argmin()
                    x = X[IDX, :]
                else:
                    raise ValueError("Either 'index' or 'weights' must be selected in the dialog")

            self.update_airfoil_parameters_from_vector(x)

    def export_design_variable_values(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = SaveAsDialog(self, file_filter=file_filter)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            parameter_list = self.geo_col.extract_design_variable_values(bounds_normalized=True)
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
        for child in self.airfoil_canvas.plot.allChildItems():
            if isinstance(child, pg.PColorMeshItem) or isinstance(child, PymeadPColorMeshItem):
                self.airfoil_canvas.plot.getViewBox().removeItem(child)
        if self.cbar is not None:
            self.airfoil_canvas.plot.getViewBox().removeItem(self.cbar)
            self.cbar = None
            self.cbar_label_attrs = None

    def plot_field(self):
        if self.last_analysis_dir is None and get_setting("plot-field-dir") != "":
            default_field_dir = get_setting("plot-field-dir")
        elif self.last_analysis_dir is not None:
            default_field_dir = self.last_analysis_dir
        else:
            default_field_dir = ""
        dlg = MSESFieldPlotDialog(parent=self, default_field_dir=default_field_dir,
                                  theme=self.themes[self.current_theme])
        if dlg.exec_():
            inputs = dlg.valuesFromWidgets()
        else:
            return

        self.clear_field()

        for child in self.airfoil_canvas.plot.allChildItems():
            if hasattr(child, 'setZValue'):
                child.setZValue(1.0)

        analysis_dir = inputs['analysis_dir']
        vBox = self.airfoil_canvas.plot.getViewBox()
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

        for child in self.airfoil_canvas.plot.allChildItems():
            if hasattr(child, 'setZValue') and not isinstance(child, pg.PColorMeshItem) \
                    and not isinstance(child, PymeadPColorMeshItem):
                child.setZValue(5)

        all_levels = np.array([list(pcmi.getLevels()) for pcmi in pcmi_list])
        levels = (np.min(all_levels[:, 0]), np.max(all_levels[:, 1]))
        # bar = pg.ColorBarItem(
        #     values=levels,
        #     colorMap='CET-R1',
        #     rounding=0.001,
        #     limits=levels,
        #     orientation='v',
        #     pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        # )
        # bar.setImageItem(pcmi_list)

        # bar.setLabel(**self.cbar_label_attrs)
        # self.cbar = bar
        if self.current_theme == "dark":
            col_data = cm.berlin.colors
        elif self.current_theme == "light":
            col_data = cm.vik.colors
        else:
            raise ValueError("Could not find color map for the current theme")

        if inputs["flow_variable"] == "M":
            Mach_stop = (1.0 - levels[0]) / (levels[1] - levels[0])
            pos = np.linspace(0.0, Mach_stop, col_data.shape[0] // 2 + 1)
            pos = pos[:-1]
            pos2 = np.linspace(Mach_stop, 1.0, col_data.shape[0] // 2)
            for p in pos2:
                pos = np.append(pos, p)
        elif inputs["flow_variable"] == "Cp":
            Cp_stop = (0.0 - levels[0]) / (levels[1] - levels[0])
            pos = np.linspace(0.0, Cp_stop, col_data.shape[0] // 2 + 1)
            pos = pos[:-1]
            pos2 = np.linspace(Cp_stop, 1.0, col_data.shape[0] // 2)
            for p in pos2:
                pos = np.append(pos, p)
        else:
            stop = (1.0 - levels[0]) / (levels[1] - levels[0])
            pos = np.linspace(0.0, stop, col_data.shape[0] // 2 + 1)
            pos = pos[:-1]
            pos2 = np.linspace(stop, 1.0, col_data.shape[0] // 2)
            for p in pos2:
                pos = np.append(pos, p)

        self.crameri_cmap = {
            "dark": pg.ColorMap(name="berlin", pos=pos, color=255 * cm.berlin.colors[:, :3] + 0.5),
            "light": pg.ColorMap(name="vik", pos=pos, color=255 * cm.vik.colors[:, :3] + 0.5)
        }

        self.cbar_label_attrs = {
            "axis": "right",
            "text": flow_var_label[inputs["flow_variable"]],
            "font-size": f"{get_setting('axis-label-point-size')}pt",
            "font-family": f"{get_setting('axis-label-font-family')}"
        }

        theme = self.themes[self.current_theme]

        self.cbar = self.airfoil_canvas.plot.addColorBar(pcmi_list, colorMap=self.crameri_cmap[self.current_theme])
        self.cbar_label_attrs["color"] = theme["cbar-color"]

        self.cbar.setLabel(**self.cbar_label_attrs)
        self.cbar.setLevels(values=levels)

        tick_font = QFont(get_setting("cbar-tick-font-family"), get_setting("cbar-tick-point-size"))
        self.cbar.axis.setStyle(tickFont=tick_font)
        self.cbar.axis.setTextPen(pg.mkPen(color=theme["cbar-color"]))
        self.cbar.getAxis("right").setWidth(20 + 2 * get_setting("axis-label-point-size") +
                                            2 * get_setting("cbar-tick-point-size"))

    def load_geo_col_no_dialog(self, file_name):
        self.permanent_widget.progress_bar.setValue(0)
        self.permanent_widget.progress_bar.show()
        self.statusBar().showMessage("Loading MEA...")
        n_func_strs = count_func_strs(file_name)

        self.airfoil_canvas.clear()
        self.parameter_tree.clear()
        self.parameter_tree.addContainers()

        self.permanent_widget.progress_bar.setValue(10)
        self.statusBar().showMessage("Adding airfoils...")
        # for a in self.mea.airfoils.values():
        #     a.update()
        # self.v.clear()
        # self.param_tree_instance.t.clear()
        geo_col_dict = load_data(file_name)
        self.geo_col = GeometryCollection.set_from_dict_rep(geo_col_dict, canvas=self.airfoil_canvas,
                                                            tree=self.parameter_tree, gui_obj=self)

        self.permanent_widget.progress_bar.setValue(20)
        # for idx, airfoil in enumerate(self.mea.airfoils.values()):
        #     self.mea.add_airfoil_graph_to_airfoil(airfoil, idx, None, w=self.w, v=self.v, gui_obj=self)
        self.permanent_widget.progress_bar.setValue(25)
        ProgressInfo = namedtuple("ProgressInfo", ("start", "end", "n"))
        progress_info = ProgressInfo(25, 85, n_func_strs)
        # self.param_tree_instance = MEAParamTree(self.mea, self.statusBar(), parent=self, progress_info=progress_info)
        self.permanent_widget.progress_bar.setValue(85)
        # for a in self.mea.airfoils.values():
        #     a.airfoil_graph.param_tree = self.param_tree_instance
        #     a.airfoil_graph.airfoil_parameters = a.airfoil_graph.param_tree.p.param('Airfoil Parameters')
        # dben = benedict.benedict(self.mea.param_dict)
        # self.progress_bar.setValue(90)
        # for k in dben.keypaths():
        #     param = dben[k]
        #     if isinstance(param, Param):
        #         if param.mea is None:
        #             param.mea = self.mea
        #         if param.mea.param_tree is None:
        #             param.mea.param_tree = self.param_tree_instance
        # self.mea.param_tree = self.param_tree_instance
        # self.design_tree_widget = self.param_tree_instance.t
        # widget0 = self.main_layout.itemAt(0).widget()
        # self.main_layout.replaceWidget(widget0, self.design_tree_widget)
        # widget0.deleteLater()
        # self.mea_start_dict = self.copy_mea_dict()
        self.permanent_widget.progress_bar.setValue(100)
        self.statusBar().showMessage("Airfoil system load complete.", 2000)
        self.permanent_widget.progress_bar.hide()
        self.geo_col.tree.geo_col = self.geo_col
        self.geo_col.canvas.geo_col = self.geo_col
        self.auto_range_geometry()

    def disp_message_box(self, message: str, message_mode: str = 'error', rich_text: bool = False):
        disp_message_box(message, self, message_mode=message_mode, rich_text=rich_text)

    def output_area_text(self, text: str, mode: str = 'plain', mono: bool = True, line_break: bool = False):
        # prepend_html = f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style>" \
        #                f"</head><body><p><font size='20pt'>&#8203;</font></p></body>"
        previous_cursor = self.text_area.textCursor()
        self.text_area.moveCursor(QTextCursor.End)
        if mode == 'plain':
            # if mode == "plain" and mono:
            #     self.text_area.insertHtml(prepend_html)
            # self.text_area.insertPlainText(text)
            line_break = "<br>" if line_break else ""

            # Note: the "pre" tag prevents whitespace collapse
            self.text_area.insertHtml(f'<body><pre><p>{text}{line_break}</p></pre></body>')
        elif mode == 'html':
            self.text_area.insertHtml(text)
        else:
            raise ValueError('Mode must be \'plain\' or \'html\'')
        self.text_area.setTextCursor(previous_cursor)
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def showColoredMessage(self, message: str, msecs: int, color: str):
        self.status_bar.setStyleSheet(f"color: {color}")
        self.permanent_widget.info_label.setStyleSheet(f"color: {self.themes[self.current_theme]['main-color']}")
        self.status_bar.showMessage(message, msecs)

    def statusChanged(self, args):
        if not args:
            self.status_bar.setStyleSheet(f"color: {self.themes[self.current_theme]['main-color']}")

    def output_link_text(self, text: str, link: str, line_break: bool = False):
        previous_cursor = self.text_area.textCursor()
        self.text_area.moveCursor(QTextCursor.End)
        line_break = "<br>" if line_break else ""
        self.text_area.insertHtml(f'<head><style> ')

    def display_airfoil_statistics(self):
        airfoil_stats = AirfoilStatistics(geo_col=self.geo_col)
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
                                           ds_curve_exp=inputs["downsampling_curve_exp"]) if inputs[
                "use_downsampling"] else {}

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
        self.dialog = ExportIGESDialog(parent=self, theme=self.themes[self.current_theme])
        if self.dialog.exec_():
            inputs = self.dialog.valuesFromWidgets()
            iges_file_path = self.geo_col.write_to_iges(base_dir=inputs["dir"], file_name=inputs["file_name"],
                                                        translation=inputs["translation"], scaling=inputs["scaling"],
                                                        rotation=inputs["rotation"],
                                                        transformation_order=inputs["transformation_order"])
            self.disp_message_box(f"Airfoil geometry saved to {iges_file_path}", message_mode="info")

    def single_airfoil_viscous_analysis(self):
        self.dialog = XFOILDialog(parent=self, current_airfoils=[k for k in self.geo_col.container()["airfoils"]],
                                  theme=self.themes[self.current_theme])
        current_airfoils = [k for k in self.geo_col.container()["airfoils"].keys()]
        self.dialog.w.widget_dict["airfoil"]["widget"].addItems(current_airfoils)
        if self.dialog.exec():
            inputs = self.dialog.valuesFromWidgets()
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

            # TODO: insert downsampling step here

            # coords = tuple(self.mea.deepcopy().airfoils[xfoil_settings['airfoil']].get_coords(
            #     body_fixed_csys=False, as_tuple=True))

            coords = self.geo_col.container()["airfoils"][xfoil_settings["airfoil"]].get_coords_selig_format()

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
                    f"[{str(self.n_analyses).zfill(2)}] ")
                self.output_area_text(
                    f"<a href='file:///{os.path.join(xfoil_settings['airfoil_analysis_dir'], xfoil_settings['airfoil_coord_file_name'])}'><font family='DejaVu Sans Mono' size='3'>XFOIL</font></a>",
                    mode="html")
                self.output_area_text(
                    f" Converged = {aero_data['converged']} | Errored out = "
                    f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}", line_break=True)
            else:
                self.output_area_text(
                    f"[{str(self.n_analyses).zfill(2)}] ")
                self.output_area_text(
                    f"<a href='file:///{os.path.join(xfoil_settings['airfoil_analysis_dir'], xfoil_settings['airfoil_coord_file_name'])}'><font family='DejaVu Sans Mono' size='3'>XFOIL</font></a>",
                    mode="html")
                self.output_area_text(f" ({xfoil_settings['airfoil']}, "
                                      f"\u03b1 = {aero_data['alf']:.3f}, Re = {xfoil_settings['Re']:.3E}, "
                                      f"Ma = {xfoil_settings['Ma']:.3f}): "
                                      f"Cl = {aero_data['Cl']:+7.4f} | Cd = {aero_data['Cd']:+.5f} | Cm = {aero_data['Cm']:+7.4f} "
                                      f"| L/D = {aero_data['L/D']:+8.4f}".replace("-", "\u2212"), line_break=True)
            bar = self.text_area.verticalScrollBar()
            sb = bar
            sb.setValue(sb.maximum())

            if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
                if self.analysis_graph is None:
                    # Need to set analysis_graph to None if analysis window is closed! Might also not want to allow geometry docking window to be closed
                    self.analysis_graph = AnalysisGraph(
                        background_color=self.themes[self.current_theme]["graph-background-color"])
                    self.add_new_tab_widget(self.analysis_graph.w, "Analysis")
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

        self.dialog = MultiAirfoilDialog(
            parent=self, geo_col=self.geo_col, theme=self.themes[self.current_theme],
            settings_override=self.multi_airfoil_analysis_settings
        )
        self.dialog.accepted.connect(self.multi_airfoil_analysis_accepted)
        self.dialog.rejected.connect(self.multi_airfoil_analysis_rejected)
        self.dialog.exec_()

    def multi_airfoil_analysis_accepted(self):

        inputs = self.dialog.valuesFromWidgets()
        self.multi_airfoil_analysis_settings = inputs

        if inputs is not None:
            mset_settings = convert_dialog_to_mset_settings(inputs['MSET'])
            mses_settings = convert_dialog_to_mses_settings(inputs['MSES'])
            mplot_settings = convert_dialog_to_mplot_settings(inputs['MPLOT'])
            self.multi_airfoil_analysis(mset_settings, mses_settings, mplot_settings)

    def multi_airfoil_analysis_rejected(self):
        self.multi_airfoil_analysis_settings = self.dialog.valuesFromWidgets()

    def display_mses_result(self, aero_data: dict, mset_settings: dict, mses_settings: dict):

        def display_fail():
            # Throw a GUI error
            self.disp_message_box("MSES Analysis Failed", message_mode='error')

            # Output failed MSES analysis info to console
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] "
                f"MSES Converged = {aero_data['converged']} | "
                f"Errored out = {aero_data['errored_out']} | "
                f"Timed out = {aero_data['timed_out']}",
                line_break=True
            )

        def display_success():
            # Calculate L/D if necessary
            if "L/D" not in aero_data.keys():
                aero_data["L/D"] = np.true_divide(aero_data["Cl"], aero_data["Cd"])

            # Compute the output analysis directory
            analysis_dir_full_path = os.path.abspath(
                os.path.join(mset_settings['airfoil_analysis_dir'], mset_settings['airfoil_coord_file_name'], '')
            )

            # Output successful MSES analysis data to console
            self.output_area_text(f"[{str(self.n_analyses).zfill(2)}] ")  # Number of analyses
            self.output_area_text(f"<a href='{analysis_dir_full_path}'>MSES</a>", mode="html")  # Folder link
            self.output_area_text(
                f" ({mset_settings['mea']}, "
                f"\u03b1 = {aero_data['alf']:.3f}\u00b0, "
                f"Re = {mses_settings['REYNIN']:.3E}, "  # Reynolds number
                f"Ma = {mses_settings['MACHIN']:.3f}): "  # Mach number
                f"Cl = {aero_data['Cl']:+7.4f} | "  # Lift coefficient
                f"Cd = {aero_data['Cd']:+.5f} | "  # Drag coefficient
                f"Cm = {aero_data['Cm']:+7.4f} | "  # Pitching moment coefficient
                f"L/D = {aero_data['L/D']:+8.4f}".replace("-", "\u2212"),  # Lift-to-drag ratio
                # Note that the hyphens are replaced with the unicode subtraction char because the hyphen does not
                # obey the fixed-width rule
                line_break=True
            )

        if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
            display_fail()
        else:
            display_success()

    def display_svgs(self, mset_settings: dict, mplot_settings: dict):

        def display_svg():
            f_name = os.path.join(mset_settings['airfoil_analysis_dir'],
                                  mset_settings['airfoil_coord_file_name'],
                                  f"{svg_plot}.svg")
            if not os.path.exists(f_name):
                self.disp_message_box(f"SVG file {f_name} was not found")
                return

            image = QSvgWidget(f_name)
            graphics_scene = QGraphicsScene()
            graphics_scene.addWidget(image)
            view = CustomGraphicsView(graphics_scene, parent=self)
            view.setRenderHint(QPainter.Antialiasing)
            Mach_contour_widget = QWidget(self)
            widget_layout = QGridLayout()
            Mach_contour_widget.setLayout(widget_layout)
            widget_layout.addWidget(view, 0, 0, 4, 4)
            start_counter = 1
            max_tab_name_search = 1000
            for idx in range(max_tab_name_search):
                name = f"{svg_plot}_{start_counter}"
                if name in self.dock_widget_names:
                    start_counter += 1
                else:
                    self.add_new_tab_widget(Mach_contour_widget, name)
                    break

        for svg_plot in SVG_PLOTS:
            if mplot_settings[SVG_SETTINGS_TR[svg_plot]]:
                display_svg()

    def plot_mses_pressure_coefficient_distribution(self, aero_data: dict, mea: MEA):
        if self.analysis_graph is None:
            # Need to set analysis_graph to None if analysis window is closed
            self.analysis_graph = AnalysisGraph(
                background_color=self.themes[self.current_theme]["graph-background-color"])
            self.add_new_tab_widget(self.analysis_graph.w, "Analysis")

        # Get the index of the pen to determine which style to use
        pen_idx = self.n_converged_analyses % len(self.pens)

        # Get the maximum physical extent of the airfoil system in the x-direction (used to prevent showing
        # off-body pressure recovery)
        x_max = mea.get_max_x_extent()

        # Plot the Cp distribution for each airfoil side
        for side in aero_data["BL"]:
            pg_plot_handle = self.analysis_graph.v.plot(
                pen=pg.mkPen(color=self.pens[pen_idx][0], style=self.pens[pen_idx][1]), name=str(self.n_analyses)
            )
            x = side["x"] if isinstance(side["x"], np.ndarray) else np.array(side["x"])
            Cp = side["Cp"] if isinstance(side["Cp"], np.ndarray) else np.array(side["Cp"])
            pg_plot_handle.setData(x[np.where(x <= x_max)[0]], Cp[np.where(x <= x_max)[0]])

    def multi_airfoil_analysis(self, mset_settings: dict, mses_settings: dict,
                               mplot_settings: dict):

        mea = self.geo_col.container()["mea"][mset_settings["mea"]]

        try:
            aero_data, _ = calculate_aero_data(mset_settings['airfoil_analysis_dir'],
                                               mset_settings['airfoil_coord_file_name'],
                                               mea=mea,
                                               tool="MSES",
                                               export_Cp=True,
                                               mset_settings=mset_settings,
                                               mses_settings=mses_settings,
                                               mplot_settings=mplot_settings)
        except OSError as os_error:
            self.disp_message_box(str(os_error), message_mode="error")
            return

        self.display_mses_result(aero_data, mset_settings, mses_settings)

        if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
            self.plot_mses_pressure_coefficient_distribution(aero_data, mea)
            self.display_svgs(mset_settings, mplot_settings)

            # Update the last successful analysis directory (for easy access in field plotting)
            self.last_analysis_dir = os.path.join(mset_settings["airfoil_analysis_dir"],
                                                  mset_settings["airfoil_coord_file_name"])

            # Increment the number of converged analyses and the total number of analyses
            self.n_converged_analyses += 1
            self.n_analyses += 1
        else:
            self.n_analyses += 1

    def match_airfoil(self):
        airfoil_names = [a for a in self.geo_col.container()["airfoils"].keys()]
        dialog = AirfoilMatchingDialog(self, airfoil_names=airfoil_names, theme=self.themes[self.current_theme])
        if dialog.exec_():
            airfoil_match_settings = dialog.valuesFromWidgets()
            # res = match_airfoil_ga(self.mea, target_airfoil, airfoil_name)
            res = match_airfoil(self.geo_col.container()["airfoils"], airfoil_match_settings["tool_airfoil"],
                                airfoil_match_settings["target_airfoil"])
            msg_mode = 'error'
            if hasattr(res, 'success') and res.success or hasattr(res, 'F') and res.F is not None:
                if hasattr(res, 'x'):
                    update_params = res.x
                elif hasattr(res, 'X'):
                    update_params = res.X
                else:
                    raise AttributeError("Did not have x or X to update airfoil parameter dictionary")
                # print(f"{update_params = }")
                self.geo_col.assign_design_variable_values(update_params, bounds_normalized=True)
                msg_mode = 'info'
            self.disp_message_box(message=res.message, message_mode=msg_mode)

    def plot_airfoil_from_airfoiltools(self):
        dialog = AirfoilPlotDialog(self, theme=self.themes[self.current_theme])
        if dialog.exec_():
            airfoil_name = dialog.valuesFromWidgets()
            airfoil = extract_data_from_airfoiltools(airfoil_name)
            self.airfoil_canvas.plot.plot(airfoil[:, 0], airfoil[:, 1], pen=pg.mkPen(color='orange', width=1))

    def setup_optimization(self):
        self.dialog = OptimizationSetupDialog(self, settings_override=self.opt_settings,
                                              geo_col=self.geo_col, theme=self.themes[self.current_theme])
        self.dialog.accepted.connect(self.optimization_accepted)
        self.dialog.rejected.connect(self.optimization_rejected)
        self.dialog.exec_()

    def optimization_accepted(self):
        exit_the_dialog = False
        early_return = False
        opt_settings_list = None
        param_dict_list = None
        geo_col_dict_list = None
        files = None
        while not exit_the_dialog and not early_return:
            self.opt_settings = self.dialog.valuesFromWidgets()

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
            geo_col_dict_list = []

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
                    # mea_dict = self.mea.copy_as_param_dict(deactivate_airfoil_graphs=True)
                    geo_col = self.geo_col.get_dict_rep()
                else:
                    mea_file = opt_settings['General Settings']['mea_file']
                    if not os.path.exists(mea_file):
                        self.disp_message_box('JMEA parametrization file not found', message_mode='error')
                        exit_the_dialog = True
                        early_return = True
                        continue
                    else:
                        # mea_dict = load_data(mea_file)
                        geo_col = load_data(mea_file)

                # # Generate the multi-element airfoil from the dictionary
                # mea = MEA.generate_from_param_dict(mea_dict)

                # TODO: reimplement this logic
                # norm_val_list = geo_col.extract_design_variable_values(bounds_normalized=True)
                # if isinstance(norm_val_list, str):
                #     error_message = norm_val_list
                #     self.disp_message_box(error_message, message_mode='error')
                #     exit_the_dialog = True
                #     early_return = True
                #     continue

                param_dict['n_var'] = len([k for k in geo_col["desvar"]])

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

                # for airfoil in mea.airfoils.values():
                #     airfoil.airfoil_graphs_active = False
                # mea.airfoil_graphs_active = False
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
                geo_col_dict_list.append(geo_col)
                exit_the_dialog = True

        if early_return:
            self.setup_optimization()

        if not early_return:
            for (opt_settings, param_dict, geo_col_dict) in zip(opt_settings_list, param_dict_list, geo_col_dict_list):
                # The next line is just to make sure any calls to the GUI are performed before the optimization
                self.dialog.setWidgetValuesFromDict(new_inputs=opt_settings)

                # Need to regenerate the objectives and constraints here since they contain references to
                # (non-serializable) modules which must be passed through a multiprocessing.Pipe
                new_obj_list = [obj.func_str for obj in self.objectives]
                new_constr_list = [constr.func_str for constr in self.constraints]

                # Run the shape optimization in a worker thread
                self.run_shape_optimization(param_dict, opt_settings,
                                            # mea.copy_as_param_dict(deactivate_airfoil_graphs=True),
                                            geo_col_dict,
                                            new_obj_list, new_constr_list,
                                            )

    def optimization_rejected(self):
        self.opt_settings = self.dialog.valuesFromWidgets()
        return

    @pyqtSlot(str, object)
    def progress_update(self, status: str, data: object):
        bcolor = self.themes[self.current_theme]["graph-background-color"]
        if status == "text" and isinstance(data, str):
            self.output_area_text(data, line_break=True)
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

    def run_shape_optimization(self, param_dict: dict, opt_settings: dict, geo_col_dict: dict,
                               objectives, constraints):

        self.clear_opt_plots()

        def run_cpu_bound_process():
            shape_opt_process = CPUBoundProcess(
                shape_optimization_static,
                args=(param_dict, opt_settings, geo_col_dict, objectives, constraints)
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

        self.shape_opt_process.terminate()

        if self.opt_thread is not None:
            self.opt_thread.join()

        self.shape_opt_process = None

        print("Optimization terminated.")

    @staticmethod
    def generate_output_folder_link_text(folder: str):
        return f"<a href='{folder}' style='font-family:DejaVu Sans Mono; " \
               f"color: #1FBBCC; font-size: 10pt;'>Open output folder</a><br>"

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
        else:
            self.showNormal()

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_Escape:
            self.geo_col.clear_selected_objects()
            self.status_bar.clearMessage()
        if a0.key() == Qt.Key_Delete:
            self.geo_col.remove_selected_objects()
            self.status_bar.clearMessage()


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
