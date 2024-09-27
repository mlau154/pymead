import multiprocessing as mp
import os
import shutil
import sys
import typing
import warnings
from copy import deepcopy
from functools import partial
from threading import Thread
from pathlib import PureWindowsPath

import networkx
import numpy as np
import pyqtgraph as pg
import requests
from scipy.optimize import OptimizeResult
from PyQt6.QtCore import QEvent, QObject, Qt, QThreadPool, QRect
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QIcon, QFont, QFontDatabase, QPainter, QCloseEvent, QTextCursor, QImage, QAction, QColor
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import QApplication, \
    QWidget, QMenu, QStatusBar, QGraphicsScene, QGridLayout, QDockWidget, QSizeGrip

from pymoo.decomposition.asf import ASF
from pyqtgraph.exporters import CSVExporter, SVGExporter
from qframelesswindow import FramelessMainWindow

from pymead import ICON_DIR, GUI_SETTINGS_DIR, GUI_THEMES_DIR, RESOURCE_DIR, EXAMPLES_DIR, q_settings, \
    TargetPathNotFoundError
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
from pymead.gui.analysis_graph import AnalysisGraph, ResidualGraph, PolarGraphCollection, \
    AirfoilMatchingGraphCollection, ResourcesGraph
from pymead.gui.concurrency import CPUBoundProcess
from pymead.gui.custom_graphics_view import CustomGraphicsView
from pymead.gui.dockable_tab_widget import PymeadDockWidget
from pymead.gui.help_browser import HelpBrowserWindow
from pymead.gui.dialogs import LoadDialog, SaveAsDialog, OptimizationSetupDialog, \
    MultiAirfoilDialog, ExportCoordinatesDialog, ExportControlPointsDialog, \
    AirfoilMatchingDialog, MSESFieldPlotDialog, ExportIGESDialog, XFOILDialog, NewGeoColDialog, EditBoundsDialog, \
    ExitDialog, ScreenshotDialog, LoadAirfoilAlgFile, ExitOptimizationDialog, SettingsDialog, LoadPointsDialog, \
    PanelDialog, FileOverwriteDialog
from pymead.gui.dialogs import convert_dialog_to_mset_settings, convert_dialog_to_mses_settings, \
    convert_dialog_to_mplot_settings, convert_dialog_to_mpolar_settings, convert_opt_settings_to_param_dict
from pymead.gui.main_icon_toolbar import MainIconToolbar
from pymead.gui.menu_action import MenuAction
from pymead.gui.message_box import disp_message_box
from pymead.gui.parameter_tree import ParameterTree
from pymead.gui.permanent_widget import PermanentWidget
from pymead.gui.show_hide import ShowHideDialog
from pymead.gui.side_grip import SideGrip
from pymead.gui.text_area import ConsoleTextArea
from pymead.gui.title_bar import TitleBar
from pymead.gui.opt_callback import PlotAirfoilCallback, ParallelCoordsCallback, OptCallback, \
    DragPlotCallbackXFOIL, CpPlotCallbackXFOIL, DragPlotCallbackMSES, CpPlotCallbackMSES, TextCallback
from pymead.optimization.opt_setup import calculate_warm_start_index
from pymead.optimization.opt_setup import read_stencil_from_array
from pymead.optimization.resources import display_resources
from pymead.optimization.shape_optimization import shape_optimization as shape_optimization_static
from pymead.post.mses_field import flow_var_label
from pymead.optimization.airfoil_matching import match_airfoil
from pymead.utils.dict_recursion import compare_dicts_floating_precision
from pymead.utils.get_airfoil import extract_data_from_airfoiltools, AirfoilNotFoundError
from pymead.utils.misc import get_setting
from pymead.utils.misc import make_ga_opt_dir
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.version_check import using_latest

q_settings_descriptions = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))

# Suppress the following DeprecationWarning: sipPyTypeDict() is deprecated, the extension module
# should use sipPyTypeDictRef() instead
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GUI(FramelessMainWindow):
    _gripSize = 5

    def __init__(self,
                 path=None,
                 parent=None,
                 bypass_vercheck: bool = False,
                 bypass_exit_save_dialog: bool = False
                 ):
        # try:
        #     import pyi_splash
        #     pyi_splash.update_text("Initializing constants...")
        # except:
        #     pass
        super().__init__(parent=parent)
        self.bypass_exit_save_dialog = bypass_exit_save_dialog
        self.showHideState = None
        self.windowMaximized = False
        self.pool = None
        self.current_opt_folder = None

        # Set up MSES process
        self.mses_thread = None
        self.mses_process = None

        # Set up optimization process
        self.cpu_bound_process = None
        self.opt_thread = None
        self.shape_opt_process = None

        # Set up match airfoil process
        self.match_airfoil_process = None
        self.match_airfoil_thread = None

        # Set up resources process
        self.display_resources_process = None
        self.display_resources_thread = None

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

        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []

        self.design_tree = None
        self.dialog = None
        self.opt_settings = None
        self.multi_airfoil_analysis_settings = None
        self.xfoil_settings = None
        self.panel_settings = None
        self.screenshot_settings = None
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
        self.field_plot_variable = None
        self.analysis_graph = None
        self.cached_cp_data = []
        self.resources_graph = None
        self.residual_graph = None
        self.residual_data = None
        self.polar_graph_collection = None
        self.airfoil_matching_graph_collection = None
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

        # Save/load variables
        self.save_attempts = 0
        self.last_saved_state = None
        self.current_save_name = None

        # mandatory for cursor updates
        self.setMouseTracking(True)

        self.title_bar = TitleBar(self, theme=self.themes[self.current_theme],
                                  window_title=self.windowTitle())
        self.title_bar.sigMessage.connect(self.disp_message_box)
        self.windowTitleChanged.connect(self.title_bar.updateTitle)
        self.titleBar.hide()

        self.sideGrips = [
            SideGrip(self, Qt.Edge.LeftEdge),
            SideGrip(self, Qt.Edge.TopEdge),
            SideGrip(self, Qt.Edge.RightEdge),
            SideGrip(self, Qt.Edge.BottomEdge),
        ]
        # corner grips should be "on top" of everything, otherwise the side grips
        # will take precedence on mouse events, so we are adding them *after*;
        # alternatively, widget.raise_() can be used
        self.cornerGrips = [QSizeGrip(self) for _ in range(4)]

        # try:
        #     pyi_splash.update_text("Generating pymead widgets...")
        # except:
        #     pass

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
        self.geo_col.units.set_current_length_unit(get_setting("length_unit"))
        self.geo_col.units.set_current_angle_unit(get_setting("angle_unit"))
        self.geo_col.units.set_current_area_unit(get_setting("area_unit"))

        self.airfoil_canvas = AirfoilCanvas(self, geo_col=self.geo_col, gui_obj=self)
        self.airfoil_canvas.sigStatusBarUpdate.connect(self.setStatusBarText)

        self.parameter_tree = ParameterTree(geo_col=self.geo_col, parent=self, gui_obj=self)

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

        self.auto_range_geometry()
        self.statusBar().clearMessage()
        self.permanent_widget = PermanentWidget(self.geo_col, self)
        self.statusBar().addPermanentWidget(self.permanent_widget)
        self.permanent_widget.updateAirfoils()

        self.parameter_tree.setMaximumWidth(300)
        self.resize(1000, self.title_bar.height() + 600)
        self.parameter_tree.setMaximumWidth(800)

        # try:
        #     pyi_splash.update_text("Loading geometry information...")
        # except:
        #     pass

        # Load the airfoil system from the system argument variable if necessary
        if self.path is not None:
            self.load_geo_col_no_dialog(self.path)
        else:
            self.last_saved_state = self.get_geo_col_state()

        # try:
        #     pyi_splash.close()
        # except:
        #     pass

        # Check if we are using the most recent release of pymead (notify if not)
        if bypass_vercheck:
            return
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
                                  f"https://github.com/mlau154/pymead/releases</a>. If you are using pymead directly"
                                  f" via Python, you can update with 'pip install pymead --upgrade' in the terminal.",
                                  message_mode="info", rich_text=True)

    def add_new_tab_widget(self, widget, name):
        if not (name in self.dock_widget_names):
            dw = PymeadDockWidget(name, self)
            dw.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dw.setWidget(widget)
            dw.setFloating(False)
            if name in ["Tree", "Geometry", "Console"]:
                dw.setFeatures(dw.features() & ~QDockWidget.DockWidgetFeature.DockWidgetClosable)
            dw.tab_closed.connect(self.on_tab_closed)
            self.dock_widgets.append(dw)
            self.dock_widget_names.append(name)
            if len(self.dock_widgets) == 2:
                self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_widgets[-2])  # Left
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)  # Right
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.Orientation.Horizontal)
            elif len(self.dock_widgets) == 3:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.Orientation.Vertical)
            elif len(self.dock_widgets) == 4:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-3], self.dock_widgets[-1])
            elif len(self.dock_widgets) > 4:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])

    def switch_to_tab(self, tab_name: str):
        if tab_name not in self.dock_widget_names:
            return
        self.dock_widgets[self.dock_widget_names.index(tab_name)].show()
        self.dock_widgets[self.dock_widget_names.index(tab_name)].raise_()

    def on_tab_closed(self, name: str, event: QCloseEvent):
        if name == "Analysis":
            self.analysis_graph = None
            self.n_converged_analyses = 0
        elif name == "Residuals":
            self.residual_graph = None
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
        ----------
        a0: QCloseEvent
            Qt CloseEvent object
        """
        if self.shape_opt_process is not None:
            dialog = ExitOptimizationDialog(self, theme=self.themes[self.current_theme])
            if dialog.exec():
                self.stop_process()
            else:
                a0.ignore()
                return

        # Only run the save/exit dialogs if changes have been made and not running from a test
        if not self.changes_made() or self.bypass_exit_save_dialog:
            return

        save_dialog = NewGeoColDialog(theme=self.themes[self.current_theme], parent=self)
        exit_dialog = ExitDialog(theme=self.themes[self.current_theme], parent=self)
        while True:
            if save_dialog.exec():  # If "Yes" to "Save Changes,"
                if save_dialog.save_successful:  # If the changes were saved successfully, close the program.
                    return
                if exit_dialog.exec():  # Otherwise, If "Yes" to "Exit the Program Anyway," close the program.
                    return
                a0.ignore()
                return
            # If "Cancel" to "Save Changes," end the CloseEvent and keep the program running.
            a0.ignore()
            return

    def changeEvent(self, a0):
        if a0.type() == QEvent.Type.WindowStateChange:
            self.title_bar.windowStateChanged(self.windowState())
            self.windowMaximized = self.windowState() == Qt.WindowState.WindowMaximized

            # Disable the resize grips when the window is maximized and enable them otherwise
            if self.windowMaximized:
                for grip in self.sideGrips:
                    grip.setDisabled(True)
            else:
                for grip in self.sideGrips:
                    grip.setEnabled(True)

            super().changeEvent(a0)
            a0.accept()

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
        self.status_bar.setStyleSheet(f"""color: {theme['main-color']}""")

        # Set title bar buttons
        self.title_bar.normalButton.setIcon(QIcon(os.path.join(ICON_DIR, f"normal-{theme_name}-mode.png")))
        self.title_bar.minimizeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"minimize-{theme_name}-mode.png")))
        self.title_bar.maximizeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"maximize-{theme_name}-mode.png")))
        self.title_bar.closeButton.setIcon(QIcon(os.path.join(ICON_DIR, f"close-{theme_name}-mode.png")))

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
        closed_arrow_image = os.path.join(ICON_DIR, f"closed-arrow-{self.current_theme}.png")
        opened_arrow_image = os.path.join(ICON_DIR, f"opened-arrow-{self.current_theme}.png")
        if os.path.sep == "\\":
            closed_arrow_image = PureWindowsPath(closed_arrow_image).as_posix()
            opened_arrow_image = PureWindowsPath(opened_arrow_image).as_posix()
        self.parameter_tree.setStyleSheet(
            f"""QTreeWidget::item {{ background-color: {theme['tree-background-color']}; }} 
                QTreeWidget::item:selected {{ background-color: {theme['menu-item-selected-color']}; color: {theme['main-color']} }}
                QTreeWidget::item:hover {{ color: #edb126 }}
                QTreeWidget {{ background-color: {theme['tree-background-color']} }}
                QTreeWidget::branch {{ background: {theme['tree-background-color']} }}
                QTreeWidget::branch::closed::has-children {{
                    image: url({closed_arrow_image});
                }}            

                QTreeWidget::branch::open::has-children {{
                    image: url({opened_arrow_image});
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
            self.airfoil_canvas.color_bar_data["current_theme"] = self.current_theme

        graphs_to_update = [self.analysis_graph, self.residual_graph, self.polar_graph_collection,
                            self.airfoil_matching_graph_collection, self.opt_airfoil_graph,
                            self.parallel_coords_graph, self.drag_graph, self.Cp_graph, self.resources_graph]

        for graph_to_update in graphs_to_update:
            if graph_to_update is None:
                continue
            graph_to_update.set_formatting(theme=self.themes[self.current_theme])

        for cnstr in self.geo_col.container()["geocon"].values():
            cnstr.canvas_item.setStyle(theme)

        for button_name, button_setting in self.main_icon_toolbar.button_settings.items():
            icon_name = button_setting["icon"]
            if "dark" not in icon_name:
                continue

            image_path = os.path.join(ICON_DIR, icon_name.replace("dark", theme_name))
            self.main_icon_toolbar.buttons[button_name]["button"].setIcon(QIcon(image_path))

    @staticmethod
    def pen(n: int):
        pens = [
            ('#d4251c', Qt.PenStyle.SolidLine),
            ('darkorange', Qt.PenStyle.SolidLine),
            ('gold', Qt.PenStyle.SolidLine),
            ('limegreen', Qt.PenStyle.SolidLine),
            ('cyan', Qt.PenStyle.SolidLine),
            ('mediumpurple', Qt.PenStyle.SolidLine),
            ('deeppink', Qt.PenStyle.SolidLine),
            ('#d4251c', Qt.PenStyle.DashLine),
            ('darkorange', Qt.PenStyle.DashLine),
            ('gold', Qt.PenStyle.DashLine),
            ('limegreen', Qt.PenStyle.DashLine),
            ('cyan', Qt.PenStyle.DashLine),
            ('mediumpurple', Qt.PenStyle.DashLine),
            ('deeppink', Qt.PenStyle.DashLine)
        ]
        return pens[n % len(pens)]

    def set_title_and_icon(self):
        self.setWindowTitle("pymead")
        image_path = os.path.join(ICON_DIR, "pymead-logo.png")
        self.setWindowIcon(QIcon(image_path))

    def create_menu_bar(self):
        self.menu_bar = self.menuBar()
        self.menu_bar.setNativeMenuBar(False)
        menu_data = load_data(os.path.join(GUI_SETTINGS_DIR, "menu.json"))

        def recursively_add_menus(menu: dict, menu_bar: QObject):
            for key, val in menu.items():
                if isinstance(val, dict):
                    menu = QMenu(key, parent=menu_bar)
                    menu_bar.addMenu(menu)
                    recursively_add_menus(val, menu_bar.children()[-1])
                else:
                    if isinstance(val, list) and len(val) > 2 and val[2] is not None:
                        action = QAction(key, parent=menu_bar)
                    else:
                        # QAction.triggered always emits the "checked" argument even if the action is not checkable,
                        # so use a custom class here to override this behavior.
                        action = MenuAction(key, parent=menu_bar)
                    action_parent = action.parent()
                    if isinstance(action_parent, QMenu):
                        action_parent.addAction(action)
                    else:
                        raise ValueError('Attempted to add QAction to an object not of type QMenu')
                    if isinstance(val, list):
                        shortcut_condition = val[1] is not None
                        checkable_condition = len(val) > 2 and val[2] is not None
                        extra_args_condition = len(val) > 3 and val[3] is not None
                        if extra_args_condition:
                            action.triggered.connect(partial(getattr(self, val[0]), *val[3]))
                        else:
                            if checkable_condition:
                                action.triggered.connect(getattr(self, val[0]))
                            else:
                                action.menuActionClicked.connect(getattr(self, val[0]))
                        if shortcut_condition:
                            action.setShortcut(val[1])
                        if checkable_condition:
                            action.setCheckable(True)
                            action.setChecked(val[2])
                    else:
                        if isinstance(action, MenuAction):
                            # QAction.triggered always emits the "checked" argument even if the action is not checkable,
                            # so connect the custom signal of the MenuAction here instead of triggered.
                            action.menuActionClicked.connect(getattr(self, val))
                        else:
                            # If the QAction is checkable, connect the default "triggered" signal instead. This
                            # requires that "checked" is the first positional argument of the connected method.
                            action.triggered.connect(getattr(self, val))

        recursively_add_menus(menu_data, self.menu_bar)

    def open_settings(self):
        dialog = SettingsDialog(parent=self, geo_col=self.geo_col, theme=self.themes[self.current_theme],
                                settings_override=None)
        dialog.exec()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(deepcopy(self.geo_col.get_dict_rep()))
        self.load_geo_col_from_memory(self.undo_stack[-1])
        self.undo_stack.pop()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(deepcopy(self.geo_col.get_dict_rep()))
        self.load_geo_col_from_memory(self.redo_stack[-1])
        self.redo_stack.pop()

    def load_points(self, dialog_test_action: typing.Callable = None, error_dialog_action: typing.Callable = None):
        """
        Loads a set of :math:`x`-:math:`y` points from a .dat/.txt/.csv file.
        """
        self.dialog = LoadPointsDialog(self, theme=self.themes[self.current_theme])

        if (dialog_test_action is not None and not dialog_test_action(self.dialog)) or self.dialog.exec():
            # Load the points to an array
            try:
                points = np.loadtxt(self.dialog.value())
            except ValueError:
                try:
                    points = np.loadtxt(self.dialog.value(), delimiter=",")
                except ValueError:
                    self.disp_message_box("Only whitespace and comma-delimited point files are accepted.",
                                          message_mode="error", dialog_test_action=error_dialog_action)
                    return
            except FileNotFoundError:
                self.disp_message_box(f"Could not find point file {self.dialog.value()}",
                                      message_mode="error", dialog_test_action=error_dialog_action)
                return

            # Add each point from the array to the GeometryCollection
            for point in points:
                self.geo_col.add_point(point[0], point[1])

            # Make sure that all the points are in view
            self.auto_range_geometry()

    def take_screenshot(self, dialog_test_action: typing.Callable = None,
                        info_dialog_action: typing.Callable = None,
                        error_dialog_action: typing.Callable = None) -> str:

        id_dict = {dw.windowTitle(): dw.winId() for dw in self.dock_widgets}
        id_dict = {"Full Window": self.winId(), **id_dict}

        window_widget_dict = {dw.windowTitle(): dw for dw in self.dock_widgets}
        window_widget_dict = {"Full Window": self, **window_widget_dict}

        self.dialog = ScreenshotDialog(self, theme=self.themes[self.current_theme], windows=[k for k in id_dict.keys()])
        if self.screenshot_settings is not None:
            self.dialog.setValue(self.screenshot_settings)

        final_file_name = None
        if (dialog_test_action is not None and not dialog_test_action(self.dialog)) or self.dialog.exec():
            inputs = self.dialog.value()
            self.screenshot_settings = inputs

            # Take the screenshot
            ratio = 5  # Increased device-pixel ratio for better screenshot resolution
            window_widget = window_widget_dict[inputs["window"]]
            size = window_widget.size()
            image = QImage(size.width() * ratio, size.height() * ratio, QImage.Format.Format_ARGB32)
            image.setDevicePixelRatio(ratio)
            window_widget.render(image)

            # Handle improper directory names and file extensions
            file_path_split = os.path.split(inputs['image_file'])
            dir_name = file_path_split[0]
            file_name = file_path_split[1]
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1]
            if file_ext not in [".jpg", ".jpeg"]:
                file_ext = ".jpg"

            final_file_name = os.path.join(dir_name, file_name_no_ext + file_ext)

            # Save the image or raise an error if the file directory is not found
            if os.path.isdir(dir_name):
                # screenshot.save(final_file_name, "jpg")  # Save the screenshot
                image.save(final_file_name, "jpg")  # Save the screenshot
                self.disp_message_box(f"{inputs['window']} window screenshot saved to {final_file_name}",
                                      message_mode="info", dialog_test_action=info_dialog_action)
            else:
                self.disp_message_box(f"Directory '{dir_name}' for "
                                      f"file '{file_name}' not found", dialog_test_action=error_dialog_action)

        return str(final_file_name)

    def showHidePymeadObjs(self, sub_container: str, show: bool):
        if show:
            self.airfoil_canvas.showPymeadObjs(sub_container)
        else:
            self.airfoil_canvas.hidePymeadObjs(sub_container)

    def openShowHidePymeadObjDialog(self):
        if self.showHideState is None:
            self.showAllPymeadObjs()
        dialog = ShowHideDialog(self, state=self.showHideState, theme=self.themes[self.current_theme])
        dialog.exec()

    def showAllPymeadObjs(self):
        self.showHideState = self.airfoil_canvas.showAllPymeadObjs()

    def hideAllPymeadObjs(self):
        self.showHideState = self.airfoil_canvas.hideAllPymeadObjs()

    def save_as_geo_col(self) -> bool:
        dialog = SaveAsDialog(self)
        if dialog.exec():
            current_save_name = dialog.selectedFiles()[0]
            if current_save_name[-5:] != '.jmea':
                current_save_name += '.jmea'

            # Handle the case where the file already exists and allow the user to cancel the operation if desired
            overwrite_code = 1
            if os.path.exists(current_save_name):
                self.dialog = FileOverwriteDialog(theme=self.themes[self.current_theme], parent=self)
                overwrite_code = self.dialog.exec()
            if not overwrite_code:
                return False

            self.current_save_name = current_save_name
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
            last_saved_state = self.geo_col.get_dict_rep()
            save_data(last_saved_state, self.current_save_name)
            self.last_saved_state = self.get_geo_col_state()
            self.setWindowTitle(f"pymead - {os.path.split(self.current_save_name)[-1]}")
            self.save_attempts = 0
            return True

    def deepcopy_geo_col(self):
        return deepcopy(self.geo_col)

    def load_geo_col(self, file_name: str = None):

        if self.changes_made():
            save_dialog = NewGeoColDialog(theme=self.themes[self.current_theme], parent=self,
                                          message="Airfoil has changes. Save?")
            exit_dialog = ExitDialog(theme=self.themes[self.current_theme], parent=self, window_title="Load anyway?",
                                     message="Airfoil not saved.\nAre you sure you want to load a new one?")
            while True:
                if save_dialog.exec():  # If "Yes" to "Save Changes,"
                    if save_dialog.save_successful:  # If the changes were saved successfully, close the program.
                        break
                    else:
                        if exit_dialog.exec():  # Otherwise, If "Yes" to "Exit the Program Anyway," close the program.
                            break
                    if save_dialog.reject_changes:  # If "No" to "Save Changes," do not load an MEA.
                        return
                else:  # If "Cancel" to "Save Changes," do not load an MEA
                    return

        if not file_name:
            dialog = LoadDialog(self, settings_var="jmea_default_open_location")
            if dialog.exec():
                file_name = dialog.selectedFiles()[0]
                file_name_parent_dir = os.path.dirname(file_name)
                q_settings.setValue(dialog.settings_var, file_name_parent_dir)
            else:
                file_name = None
        if file_name:
            self.load_geo_col_no_dialog(file_name)
            self.setWindowTitle(f"pymead - {os.path.split(file_name)[-1]}")
            self.current_save_name = file_name

    def new_geo_col(self):

        def load_blank_geo_col():
            self.load_geo_col_no_dialog()
            self.setWindowTitle(f"pymead")
            self.current_save_name = None

        if self.changes_made():
            dialog = NewGeoColDialog(theme=self.themes[self.current_theme], parent=self)
            if dialog.exec():
                load_blank_geo_col()
        else:
            load_blank_geo_col()

    def edit_bounds(self):
        if len(self.geo_col.container()["desvar"]) == 0:
            self.disp_message_box("No design variables present", message_mode="info")
            return
        bv_dialog = EditBoundsDialog(geo_col=self.geo_col, theme=self.themes[self.current_theme], parent=self)
        bv_dialog.show()
        bv_dialog.resizetoFit()

    def auto_range_geometry(self):
        """
        Adjusts the range of the airfoil canvas based on the rectangle that just encloses all the Points in the
        geometry collection, plus an offset. If no points are present, default bounds are chosen.

        Returns
        -------

        """
        x_data_range, y_data_range = self.airfoil_canvas.getPointRange()
        self.airfoil_canvas.plot.getViewBox().setRange(xRange=x_data_range, yRange=y_data_range)

    def import_design_variable_values(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = LoadDialog(self, settings_var="parameter_list_default_open_location", file_filter=file_filter)
        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            q_settings.setValue(dialog.settings_var, os.path.dirname(file_name))
            param_vec = np.loadtxt(file_name).tolist()
            if isinstance(param_vec, float):
                param_vec = [param_vec]
            self.geo_col.assign_design_variable_values(param_vec, bounds_normalized=True)

    def import_algorithm_pkl_file(self, dialog_test_action: typing.Callable = None):
        self.dialog = LoadAirfoilAlgFile(theme=self.themes[self.current_theme], parent=self)
        if (dialog_test_action is not None and not dialog_test_action(self.dialog)) or self.dialog.exec():
            inputs = self.dialog.valuesFromWidgets()

            if dialog_test_action is None:
                self.dialog.load_airfoil_alg_file_widget.assignQSettings(inputs)

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
                    decomp = ASF()

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

            self.geo_col.assign_design_variable_values(x, bounds_normalized=True)
            return x

    def export_design_variable_values(self):
        """This function imports a list of parameters normalized by their bounds"""
        file_filter = "DAT Files (*.dat)"
        dialog = SaveAsDialog(self, file_filter=file_filter)
        if dialog.exec():
            file_name = dialog.selectedFiles()[0]
            parameter_list = self.geo_col.extract_design_variable_values(bounds_normalized=True)
            np.savetxt(file_name, np.array(parameter_list))

    def clear_field(self):
        for child in self.airfoil_canvas.plot.allChildItems():
            if isinstance(child, pg.PColorMeshItem) or isinstance(child, pg.ColorBarItem):
                self.airfoil_canvas.plot.getViewBox().removeItem(child)
        self.cbar = None
        self.cbar_label_attrs = None
        if self.airfoil_canvas.color_bar_data is not None:
            self.airfoil_canvas.color_bar_data.clear()

    def plot_field(self, **kwargs):
        if ("show_dialog" in kwargs and kwargs["show_dialog"]) or "show_dialog" not in kwargs:
            if self.last_analysis_dir is None and get_setting("plot-field-dir") != "":
                default_field_dir = get_setting("plot-field-dir")
            elif self.last_analysis_dir is not None:
                default_field_dir = self.last_analysis_dir
            else:
                default_field_dir = ""
            dlg = MSESFieldPlotDialog(parent=self, default_field_dir=default_field_dir,
                                      theme=self.themes[self.current_theme])
            if dlg.exec():
                inputs = dlg.value()
                self.field_plot_variable = inputs["flow_variable"]
            else:
                return
        else:
            inputs = {
                "analysis_dir": self.last_analysis_dir,
                "flow_variable": self.field_plot_variable
            }

        self.clear_field()

        for child in self.airfoil_canvas.plot.allChildItems():
            if hasattr(child, 'setZValue'):
                child.setZValue(1.0)

        analysis_dir = inputs['analysis_dir']
        vBox = self.airfoil_canvas.plot.getViewBox()
        field_file = os.path.join(analysis_dir, f'field.{os.path.split(analysis_dir)[-1]}')
        grid_stats_file = os.path.join(analysis_dir, 'mplot_grid_stats.log')
        grid_file = os.path.join(analysis_dir, f'grid.{os.path.split(analysis_dir)[-1]}')
        transformation_file = os.path.join(analysis_dir, "transformation.json")
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
        try:
            transformation = load_data(transformation_file)
            x_grid = [x_grid_section / transformation["sx"][0] /
                      self.geo_col.units.convert_length_from_base(1, transformation["length_unit"]) *
                      self.geo_col.units.convert_length_from_base(1, self.geo_col.units.current_length_unit()) for
                      x_grid_section in x_grid]
            y_grid = [y_grid_section / transformation["sy"][0] /
                      self.geo_col.units.convert_length_from_base(1, transformation["length_unit"]) *
                      self.geo_col.units.convert_length_from_base(1, self.geo_col.units.current_length_unit()) for
                      y_grid_section in y_grid]
        except OSError:
            pass
        flow_var = field[flow_var_idx[inputs['flow_variable']]]

        edgecolors = None
        antialiasing = False
        # edgecolors = {'color': 'b', 'width': 1}  # May be uncommented to see edgecolor effect
        # antialiasing = True # May be uncommented to see antialiasing effect

        pcmi_list = []

        start_idx, end_idx = 0, x_grid[0].shape[1] - 1
        for flow_section_idx in range(grid_stats["numel"] + 1):
            flow_var_section = flow_var[:, start_idx:end_idx]

            pcmi = pg.PColorMeshItem(current_theme=self.current_theme,
                                     flow_var=inputs["flow_variable"], edgecolors=edgecolors,
                                     antialiasing=antialiasing)
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
            if hasattr(child, 'setZValue') and not isinstance(child, pg.PColorMeshItem):
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

        self.crameri_cmap = {}

        self.cbar_label_attrs = {
            "axis": "right",
            "text": flow_var_label[inputs["flow_variable"]],
            "font-size": f"{get_setting('axis-label-point-size')}pt",
            "font-family": f"{get_setting('axis-label-font-family')}"
        }

        theme = self.themes[self.current_theme]

        self.cbar = self.airfoil_canvas.plot.addColorBar(pcmi_list)
        self.cbar_label_attrs["color"] = theme["cbar-color"]

        self.cbar.setLabel(**self.cbar_label_attrs)
        self.airfoil_canvas.setColorBarData(cmap_dict=self.crameri_cmap, color_bar=self.cbar,
                                            current_theme=self.current_theme, flow_var=inputs["flow_variable"],
                                            min_level_default=levels[0], max_level_default=levels[1])
        self.airfoil_canvas.setColorBarLevels()

        tick_font = QFont(get_setting("cbar-tick-font-family"), get_setting("cbar-tick-point-size"))
        self.cbar.axis.setStyle(tickFont=tick_font)
        self.cbar.axis.setTextPen(pg.mkPen(color=theme["cbar-color"]))
        self.cbar.getAxis("right").setWidth(20 + 2 * get_setting("axis-label-point-size") +
                                            2 * get_setting("cbar-tick-point-size"))

    def load_geo_col_from_memory(self, dict_rep: dict):

        # Clear the canvas and the tree, and add the high-level containers back to the tree
        self.airfoil_canvas.clear()
        self.parameter_tree.clear()
        self.parameter_tree.addContainers()

        self.geo_col = GeometryCollection.set_from_dict_rep(dict_rep, canvas=self.airfoil_canvas,
                                                            tree=self.parameter_tree, gui_obj=self)
        self.permanent_widget.geo_col = self.geo_col
        self.last_saved_state = self.get_geo_col_state()

        self.geo_col.tree.geo_col = self.geo_col
        self.geo_col.canvas.geo_col = self.geo_col
        self.airfoil_canvas.setAxisLabels(self.themes[self.current_theme])
        self.permanent_widget.updateAirfoils()

    def load_geo_col_no_dialog(self, file_name: str = None):

        # Clear the canvas and the tree, and add the high-level containers back to the tree
        self.airfoil_canvas.clear()
        self.parameter_tree.clear()
        self.parameter_tree.addContainers()

        if file_name is not None:
            geo_col_dict = load_data(file_name)
        else:
            geo_col_dict = GeometryCollection().get_dict_rep()

        self.geo_col = GeometryCollection.set_from_dict_rep(geo_col_dict, canvas=self.airfoil_canvas,
                                                            tree=self.parameter_tree, gui_obj=self)
        self.permanent_widget.geo_col = self.geo_col
        self.last_saved_state = self.get_geo_col_state()

        self.geo_col.tree.geo_col = self.geo_col
        self.geo_col.canvas.geo_col = self.geo_col
        self.airfoil_canvas.setAxisLabels(self.themes[self.current_theme])
        self.permanent_widget.updateAirfoils()
        self.auto_range_geometry()

        if self.showHideState is None:
            self.showAllPymeadObjs()
        else:
            for k, v in self.showHideState.items():
                self.showHidePymeadObjs(k, v)

        # self.geo_col.switch_units("angle", old_unit=UNITS.current_angle_unit(),
        #                           new_unit=geo_col_dict["metadata"]["angle_unit"])
        # self.geo_col.switch_units("length", old_unit=UNITS.current_length_unit(),
        #                           new_unit=geo_col_dict["metadata"]["length_unit"])

    def show_constraint_graph(self):

        networkx.draw_circular(self.geo_col.gcs, labels={point: point.name() for point in self.geo_col.gcs.nodes})
        from matplotlib import pyplot as plt
        plt.show()

    def show_param_graph(self):

        networkx.draw_circular(self.geo_col.param_graph,
                               labels={param: param.name() for param in self.geo_col.param_graph.nodes})
        from matplotlib import pyplot as plt
        plt.show()

    def get_geo_col_state(self):
        return {k: v for k, v in self.geo_col.get_dict_rep().items() if k != "metadata"}

    def changes_made(self, atol: float = 1e-15) -> bool:
        return not compare_dicts_floating_precision(self.last_saved_state, self.get_geo_col_state(), atol=atol)

    def disp_message_box(self, message: str, message_mode: str = "error", rich_text: bool = False,
                         dialog_test_action: typing.Callable = None):
        """
        Displays a custom message box

        Parameters
        ----------
        message: str
            Message to display
        message_mode: str
            Type of message to send (either 'error', 'info', or 'warn')
        rich_text: bool
            Whether to display the message using rich text
        dialog_test_action: typing.Callable or None
            If not ``None``, a function pointer should be specified that intercepts the call to ``exec`` and
            performs custom actions. Used for testing only.
        """
        disp_message_box(message, self, message_mode=message_mode, rich_text=rich_text,
                         theme=self.themes[self.current_theme], dialog_test_action=dialog_test_action)

    def output_area_text(self, text: str, mode: str = 'plain', mono: bool = True, line_break: bool = False):
        # prepend_html = f"<head><style>body {{font-family: DejaVu Sans Mono;}}</style>" \
        #                f"</head><body><p><font size='20pt'>&#8203;</font></p></body>"
        previous_cursor = self.text_area.textCursor()
        self.text_area.moveCursor(QTextCursor.MoveOperation.End)
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
        self.text_area.moveCursor(QTextCursor.MoveOperation.End)
        line_break = "<br>" if line_break else ""
        self.text_area.insertHtml(f'<head><style> ')

    def display_airfoil_statistics(self):
        airfoil_stats = AirfoilStatistics(geo_col=self.geo_col)
        dialog = AirfoilStatisticsDialog(parent=self, airfoil_stats=airfoil_stats,
                                         theme=self.themes[self.current_theme])
        dialog.exec()

    def single_airfoil_inviscid_analysis(self, plot_cp: bool):
        selected_airfoil_name = self.permanent_widget.inviscid_cl_combo.currentText()

        if selected_airfoil_name == "":
            if plot_cp:
                self.disp_message_box("Choose an airfoil in the bottom right-hand corner of the screen "
                                      "to perform an incompressible, inviscid analysis", message_mode="info")
            return

        dialog = PanelDialog(self, theme=self.themes[self.current_theme], settings_override=self.panel_settings)

        if dialog.exec():
            alpha_add = dialog.value()["alfa"]
            self.panel_settings = dialog.value()
        else:
            return

        selected_airfoil = self.geo_col.container()["airfoils"][selected_airfoil_name]
        body_fixed_coords = selected_airfoil.get_chord_relative_coords()
        alpha = selected_airfoil.measure_alpha() * 180 / np.pi + alpha_add
        xy, CP, CL = single_element_inviscid(body_fixed_coords, alpha=alpha)

        if plot_cp:
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] ")
            self.output_area_text(f"PANEL ({selected_airfoil_name}, \u03b1 = {alpha:.3f}\u00b0): "
                                  f"Cl = {CL:+7.4f}".replace("-", "\u2212"), line_break=True)
        else:
            self.statusBar().showMessage(f"CL = {CL:.3f}", 4000)

        if not plot_cp:
            return

        if self.analysis_graph is None:
            self.analysis_graph = AnalysisGraph(
                theme=self.themes[self.current_theme],
                gui_obj=self,
                background_color=self.themes[self.current_theme]["graph-background-color"],
                grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
            )
            self.add_new_tab_widget(self.analysis_graph.w, "Analysis")
        name = f"[{self.n_analyses}] P ({selected_airfoil_name}, \u03b1 = {alpha:.1f}\u00b0)"
        pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pen(self.n_converged_analyses)[0],
                                                                 style=self.pen(self.n_converged_analyses)[1]),
                                                    name=name)
        pg_plot_handle.setData(xy[:, 0], CP)

        self.analysis_graph.set_legend_label_format(self.themes[self.current_theme])
        self.n_converged_analyses += 1
        self.n_analyses += 1

    def export_coordinates(self):
        """Airfoil coordinate exporter"""
        dialog = ExportCoordinatesDialog(self, theme=self.themes[self.current_theme])
        if dialog.exec():
            inputs = dialog.value()
            f_ = os.path.join(inputs['choose_dir'], inputs['file_name'])

            # Determine if output format should be JSON:
            if os.path.splitext(f_) and os.path.splitext(f_)[-1] == '.json':
                json = True
            else:
                json = False

            airfoils = inputs['airfoil_order'].split(',')

            extra_get_coords_kwargs = dict(max_airfoil_points=inputs["downsampling_max_pts"],
                                           curvature_exp=inputs["downsampling_curve_exp"]) if inputs[
                "use_downsampling"] else {}

            if json:
                coord_dict = {}
                for a in airfoils:
                    airfoil = self.geo_col.container()["airfoils"][a]
                    coords = airfoil.get_coords_selig_format(**extra_get_coords_kwargs)
                    coord_dict[a] = coords.tolist()
                save_data(coord_dict, f_)
            else:
                with open(f_, 'w') as f:
                    new_line = ""
                    if len(inputs['header']) > 0:
                        new_line = '\n'
                    f.write(f"{inputs['header']}{new_line}")
                    for idx, a in enumerate(airfoils):
                        airfoil = self.geo_col.container()["airfoils"][a]
                        coords = airfoil.get_coords_selig_format(**extra_get_coords_kwargs)
                        for coord in coords:
                            f.write(f"{coord[0]}{inputs['delimiter']}{coord[1]}\n")
                        if idx < len(airfoils) - 1:
                            f.write(f"{inputs['separator']}")
            self.disp_message_box(f"Airfoil coordinates saved to {f_}", message_mode='info')

    def export_control_points(self):
        dialog = ExportControlPointsDialog(self, theme=self.themes[self.current_theme])
        if dialog.exec():
            inputs = dialog.value()
            f_ = os.path.join(inputs['choose_dir'], inputs['file_name'])

            airfoils = inputs['airfoil_order'].split(',')

            control_point_dict = {}
            for a in airfoils:
                airfoil = self.geo_col.container()["airfoils"][a]
                control_points = []
                for c in airfoil.curves:
                    control_points.append(c.point_sequence().as_array().tolist())
                control_point_dict[a] = control_points
            save_data(control_point_dict, f_)
            self.disp_message_box(f"Airfoil control points saved to {f_}", message_mode='info')

    def show_help(self):
        HelpBrowserWindow(parent=self)

    def export_IGES(self):
        self.dialog = ExportIGESDialog(parent=self, theme=self.themes[self.current_theme])
        if self.dialog.exec():
            inputs = self.dialog.value()
            iges_file_path = self.geo_col.write_to_iges(base_dir=inputs["dir"], file_name=inputs["file_name"],
                                                        translation=inputs["translation"], scaling=inputs["scaling"],
                                                        rotation=inputs["rotation"],
                                                        transformation_order=inputs["transformation_order"])
            self.disp_message_box(f"Airfoil geometry saved to {iges_file_path}", message_mode="info")

    def single_airfoil_viscous_analysis(self, dialog_test_action: typing.Callable = None) -> dict:
        self.dialog = XFOILDialog(parent=self, current_airfoils=[k for k in self.geo_col.container()["airfoils"]],
                                  theme=self.themes[self.current_theme], settings_override=self.xfoil_settings)
        current_airfoils = [k for k in self.geo_col.container()["airfoils"].keys()]
        self.dialog.w.widget_dict["airfoil"]["widget"].addItems(current_airfoils)
        if (dialog_test_action is not None and not dialog_test_action(self.dialog)) or self.dialog.exec():
            inputs = self.dialog.value()
            self.xfoil_settings = inputs
        else:
            return {}

        xfoil_settings = {'Re': inputs['Re'],
                          'Ma': inputs['Ma'],
                          'prescribe': inputs['prescribe'],
                          'timeout': inputs['timeout'],
                          'iter': inputs['iter'],
                          'xtr': [inputs['xtr_lower'], inputs['xtr_upper']],
                          'N': inputs['N'],
                          'base_dir': inputs['base_dir'],
                          'airfoil_name': inputs['airfoil_name'],
                          'airfoil': inputs['airfoil'],
                          "visc": inputs["viscous_flag"]}
        if xfoil_settings['prescribe'] == 'Angle of Attack (deg)':
            xfoil_settings['alfa'] = inputs['alfa']
        elif xfoil_settings['prescribe'] == 'Viscous Cl':
            xfoil_settings['Cl'] = inputs['Cl']
        elif xfoil_settings['prescribe'] == 'Inviscid Cl':
            xfoil_settings['CLI'] = inputs['CLI']

        # TODO: insert downsampling step here

        # coords = tuple(self.mea.deepcopy().airfoils[xfoil_settings['airfoil']].get_coords(
        #     body_fixed_csys=False, as_tuple=True))

        if xfoil_settings["airfoil"] == "":
            self.disp_message_box("An airfoil was not chosen for analysis")
            return {}

        coords = self.geo_col.container()["airfoils"][xfoil_settings["airfoil"]].get_scaled_coords()

        try:
            aero_data, _ = calculate_aero_data(None,
                                               xfoil_settings['base_dir'],
                                               xfoil_settings['airfoil_name'],
                                               coords=coords,
                                               tool="XFOIL",
                                               xfoil_settings=xfoil_settings,
                                               export_Cp=True
                                               )
        except ValueError as e:
            self.disp_message_box(str(e))
            return {}

        if not aero_data['converged'] or aero_data['errored_out'] or aero_data['timed_out']:
            self.disp_message_box("XFOIL Analysis Failed", message_mode='error')
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] ")
            self.output_area_text(
                f"<a href='file:///{os.path.join(xfoil_settings['base_dir'], xfoil_settings['airfoil_name'])}'><font family='DejaVu Sans Mono' size='3'>XFOIL</font></a>",
                mode="html")
            self.output_area_text(
                f" Converged = {aero_data['converged']} | Errored out = "
                f"{aero_data['errored_out']} | Timed out = {aero_data['timed_out']}", line_break=True)
        else:
            self.output_area_text(
                f"[{str(self.n_analyses).zfill(2)}] ")
            self.output_area_text(
                f"<a href='file:///{os.path.join(xfoil_settings['base_dir'], xfoil_settings['airfoil_name'])}'><font family='DejaVu Sans Mono' size='3'>XFOIL</font></a>",
                mode="html")
            if xfoil_settings["visc"]:
                self.output_area_text(f" ({xfoil_settings['airfoil']}, "
                                      f"\u03b1 = {aero_data['alf']:.3f}\u00b0, Re = {xfoil_settings['Re']:.3E}, "
                                      f"Ma = {xfoil_settings['Ma']:.3f}): "
                                      f"Cl = {aero_data['Cl']:+7.4f} | Cd = {aero_data['Cd']:+.5f} | Cm = {aero_data['Cm']:+7.4f} "
                                      f"| L/D = {aero_data['L/D']:+8.4f}".replace("-", "\u2212"), line_break=True)
            else:
                self.output_area_text(f" ({xfoil_settings['airfoil']}, "
                                      f"\u03b1 = {aero_data['alf']:.3f}\u00b0, "
                                      f"Ma = {xfoil_settings['Ma']:.3f}): "
                                      f"Cl = {aero_data['Cl']:+7.4f} | Cm = {aero_data['Cm']:+7.4f}".replace("-",
                                                                                                             "\u2212"),
                                      line_break=True)
        bar = self.text_area.verticalScrollBar()
        sb = bar
        sb.setValue(sb.maximum())

        if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
            if self.analysis_graph is None:
                # TODO: Need to set analysis_graph to None if analysis window is closed! Might also not want to allow geometry docking window to be closed
                self.analysis_graph = AnalysisGraph(
                    theme=self.themes[self.current_theme],
                    gui_obj=self,
                    background_color=self.themes[self.current_theme]["graph-background-color"],
                    grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
                )
                self.add_new_tab_widget(self.analysis_graph.w, "Analysis")

            if xfoil_settings["visc"]:
                name = (
                    f"[{str(self.n_analyses)}] X ({xfoil_settings['airfoil']}, \u03b1 = {aero_data['alf']:.1f}\u00b0,"
                    f" Re = {xfoil_settings['Re']:.1E}, Ma = {xfoil_settings['Ma']:.2f})")
            else:
                name = (
                    f"[{str(self.n_analyses)}] X ({xfoil_settings['airfoil']}, \u03b1 = {aero_data['alf']:.1f}\u00b0,"
                    f" Ma = {xfoil_settings['Ma']:.2f})")

            pg_plot_handle = self.analysis_graph.v.plot(pen=pg.mkPen(color=self.pen(self.n_converged_analyses)[0],
                                                                     style=self.pen(self.n_converged_analyses)[1]),
                                                        name=name)
            self.analysis_graph.set_legend_label_format(self.themes[self.current_theme])

            self.cached_cp_data.append({
                "tool": "XFOIL",
                "index": self.n_analyses,
                "xc": aero_data["Cp"]["x"],
                "Cp": aero_data["Cp"]["Cp"]
            })

            pg_plot_handle.setData(aero_data['Cp']['x'], aero_data['Cp']['Cp'])
            # pen = pg.mkPen(color='green')
            self.n_converged_analyses += 1
            self.n_analyses += 1
        else:
            self.n_analyses += 1

        return aero_data

    def multi_airfoil_analysis_setup(self, dialog_test_action: typing.Callable = None):

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
        if (dialog_test_action is not None and not dialog_test_action(self.dialog)) or self.dialog.exec():
            pass

    def multi_airfoil_analysis_accepted(self):

        inputs = self.dialog.value()
        self.multi_airfoil_analysis_settings = inputs

        if inputs is not None:
            mset_settings = convert_dialog_to_mset_settings(inputs["MSET"])
            mses_settings = convert_dialog_to_mses_settings(inputs["MSES"])
            mplot_settings = convert_dialog_to_mplot_settings(inputs["MPLOT"])
            mpolar_settings = convert_dialog_to_mpolar_settings(inputs["MPOLAR"])
            self.run_mses(mset_settings, mses_settings, mplot_settings, mpolar_settings)

    def multi_airfoil_analysis_rejected(self):
        self.multi_airfoil_analysis_settings = self.dialog.value()

    def display_mses_result(self, aero_data: dict, mset_settings: dict, mses_settings: dict):

        def display_fail():
            # Throw a GUI error
            self.disp_message_box("MSES Analysis Failed", message_mode="error")

            # Compute the output analysis directory
            analysis_dir_full_path = os.path.abspath(
                os.path.join(mset_settings['airfoil_analysis_dir'], mset_settings['airfoil_coord_file_name'], '')
            )

            # Output failed MSES analysis info to console
            self.output_area_text(f"[{str(self.n_analyses).zfill(2)}] ")  # Number of analyses
            self.output_area_text(f"<a href='file:///{analysis_dir_full_path}'>MSES</a>", mode="html")  # Folder link
            self.output_area_text(
                f" Converged = {aero_data['converged']} | "
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
            self.output_area_text(f"<a href='file:///{analysis_dir_full_path}'>MSES</a>", mode="html")  # Folder link
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

    def display_mpolar_result(self, aero_data: dict, mset_settings: dict, mses_settings: dict):

        # Compute the output analysis directory
        analysis_dir_full_path = os.path.abspath(
            os.path.join(mset_settings['airfoil_analysis_dir'], mset_settings['airfoil_coord_file_name'], '')
        )

        self.output_area_text(f"<a href='file:///{analysis_dir_full_path}'>MPOLAR</a>", mode="html")  # Folder link
        self.output_area_text(
            f" ({mset_settings['mea']}, "
            f"Re = {mses_settings['REYNIN']:.3E}, "  # Reynolds number
            f"Ma = {mses_settings['MACHIN']:.3f})"  # Mach number
        )
        performance_params = [k for k in ["alf_ZL", "LD_max", "alf_LD_max"] if aero_data[k] is not None]
        if performance_params:
            self.output_area_text(": ")
        if aero_data["alf_ZL"] is not None:
            line_break = performance_params.index("alf_ZL") == len(performance_params) - 1
            self.output_area_text(f"\u03b1<sub>ZL</sub> = {aero_data['alf_ZL']:.3f}\u00b0".replace(
                "-", "\u2212"), mode="html")
            if line_break:
                self.output_area_text("", line_break=True)
            else:
                self.output_area_text(", ")
        if aero_data["LD_max"] is not None:
            line_break = performance_params.index("LD_max") == len(performance_params) - 1
            self.output_area_text(f"(L/D)<sub>max</sub> = {aero_data['LD_max']:.1f}".replace(
                "-", "\u2212"), mode="html")
            if line_break:
                self.output_area_text("", line_break=True)
            else:
                self.output_area_text(", ")
        if aero_data["alf_LD_max"] is not None:
            line_break = performance_params.index("alf_LD_max") == len(performance_params) - 1
            self.output_area_text(f"\u03b1 @ (L/D)<sub>max</sub> = {aero_data['alf_LD_max']:.3f}\u00b0".replace(
                "-", "\u2212"), mode="html")
            if line_break:
                self.output_area_text("", line_break=True)
            else:
                self.output_area_text(", ")

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
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
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

    def plot_mses_pressure_coefficient_distribution(self, aero_data: dict, mea: MEA, mses_settings: dict):
        if self.analysis_graph is None:
            # Need to set analysis_graph to None if analysis window is closed
            self.analysis_graph = AnalysisGraph(
                theme=self.themes[self.current_theme],
                gui_obj=self,
                background_color=self.themes[self.current_theme]["graph-background-color"],
                grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
            )
            self.add_new_tab_widget(self.analysis_graph.w, "Analysis")

        # Get the maximum physical extent of the airfoil system in the x-direction (used to prevent showing
        # off-body pressure recovery)
        x_max = mea.get_max_x_extent()

        # Plot the Cp distribution for each airfoil side
        name = (f"[{self.n_analyses}] M ({mea.name()}, \u03b1 = {aero_data['alf']:.1f}\u00b0, "
                f"Re = {mses_settings['REYNIN']:.1E}, Ma = {mses_settings['MACHIN']:.2f})")

        for side_idx, side in enumerate(aero_data["BL"]):
            extra_opts = dict(name=name) if side_idx == 0 else {}
            pg_plot_handle = self.analysis_graph.v.plot(
                pen=pg.mkPen(color=self.pen(self.n_converged_analyses)[0],
                             style=self.pen(self.n_converged_analyses)[1]), **extra_opts
            )
            x = side["x"] if isinstance(side["x"], np.ndarray) else np.array(side["x"])
            Cp = side["Cp"] if isinstance(side["Cp"], np.ndarray) else np.array(side["Cp"])
            pg_plot_handle.setData(x[np.where(x <= x_max)[0]], Cp[np.where(x <= x_max)[0]])
            self.analysis_graph.set_legend_label_format(self.themes[self.current_theme])

    def display_resources(self):

        def run_cpu_bound_process():
            self.display_resources_process = CPUBoundProcess(
                display_resources
            )
            self.display_resources_process.progress_emitter.signals.progress.connect(self.progress_update)
            self.display_resources_process.start()

        # Start running the CPU-bound process from a worker thread (separate from the main GUI thread)
        self.display_resources_thread = Thread(target=run_cpu_bound_process)
        self.display_resources_thread.start()

    def match_airfoil(self):

        def run_cpu_bound_process():

            self.match_airfoil_process = CPUBoundProcess(
                match_airfoil,
                args=(
                    self.geo_col.get_dict_rep(),
                    airfoil_match_settings["tool_airfoil"],
                    airfoil_match_settings["target_airfoil"]
                )
            )
            self.match_airfoil_process.progress_emitter.signals.progress.connect(self.progress_update)
            self.match_airfoil_process.start()

        airfoil_names = [a for a in self.geo_col.container()["airfoils"].keys()]
        dialog = AirfoilMatchingDialog(self, airfoil_names=airfoil_names, theme=self.themes[self.current_theme])
        if dialog.exec():
            try:
                airfoil_match_settings = dialog.value()
            except TargetPathNotFoundError as e:
                self.disp_message_box(f"{str(e)}")
                return
            try:
                # Start running the CPU-bound process from a worker thread (separate from the main GUI thread)
                self.match_airfoil_thread = Thread(target=run_cpu_bound_process)
                self.match_airfoil_thread.start()
            except AirfoilNotFoundError as e:
                self.disp_message_box(f"{str(e)}")
                return

    def setup_optimization(self):
        self.dialog = OptimizationSetupDialog(self, settings_override=self.opt_settings,
                                              geo_col=self.geo_col, theme=self.themes[self.current_theme],
                                              grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        self.dialog.accepted.connect(self.optimization_accepted)
        self.dialog.rejected.connect(self.optimization_rejected)
        self.dialog.exec()

        # Close all the constraint visualization widgets that have been opened
        for widget in self.dialog.constraints_widget.widget_dict["constraints"].w_dict.values():
            if widget.sub_dialog is None:
                continue
            widget.sub_dialog.close()

    def optimization_accepted(self):
        exit_the_dialog = False
        early_return = False
        opt_settings_list = None
        param_dict_list = None
        geo_col_dict_list = None
        files = None
        while not exit_the_dialog and not early_return:
            self.opt_settings = self.dialog.value()

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

                geo_col = self.geo_col.get_dict_rep()

                try:
                    param_dict = deepcopy(convert_opt_settings_to_param_dict(
                        opt_settings, len(list(geo_col["desvar"].keys()))
                    ))
                except (ValueError, FileNotFoundError) as e:
                    self.disp_message_box(str(e))
                    return

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
                self.dialog.setValue(new_inputs=opt_settings)

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
                self.display_resources()

    def optimization_rejected(self):
        self.opt_settings = self.dialog.value()
        return

    @pyqtSlot(str, object)
    def progress_update(self, status: str, data: object):
        bcolor = self.themes[self.current_theme]["graph-background-color"]
        if status == "text" and isinstance(data, str):
            self.output_area_text(data, line_break=True)
        elif status == "message" and isinstance(data, str):
            self.message_callback_fn(data)
        elif status == "disp_message_box" and isinstance(data, str):
            self.disp_message_box(data)
        elif status == "opt_progress" and isinstance(data, dict):
            callback = TextCallback(parent=self, text_list=data["text"], completed=data["completed"],
                                    warm_start_gen=data["warm_start_gen"])
            callback.exec_callback()
        elif status == "airfoil_coords" and isinstance(data, list):
            callback = PlotAirfoilCallback(parent=self, coords=data, geo_col=self.geo_col)
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "parallel_coords" and isinstance(data, tuple):
            callback = ParallelCoordsCallback(parent=self, norm_val_list=data[0], param_name_list=data[1])
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "cp_xfoil":
            callback = CpPlotCallbackXFOIL(parent=self, Cp=data)
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "cp_mses":
            callback = CpPlotCallbackMSES(parent=self, Cp=data)
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "drag_xfoil" and isinstance(data, tuple):
            callback = DragPlotCallbackXFOIL(parent=self, Cd=data[0], Cdp=data[1], Cdf=data[2])
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "drag_mses" and isinstance(data, tuple):
            callback = DragPlotCallbackMSES(parent=self, Cd=data[0], Cdp=data[1], Cdf=data[2], Cdv=data[3], Cdw=data[4])
            callback.exec_callback(theme=self.themes[self.current_theme],
                                   grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked())
        elif status == "clear_residual_plots":
            if self.residual_graph is not None:
                for plot_item in self.residual_graph.plot_items:
                    plot_item.setData([], [])
            self.residual_data = []
            self.switch_to_tab("Residuals")
        elif status == "mses_analysis_complete" and isinstance(data, tuple):
            aero_data = data[0]
            mset_settings = data[1]
            mses_settings = data[2]
            mplot_settings = data[3]
            mea_name = data[4]
            self.display_mses_result(aero_data, mset_settings, mses_settings)

            if aero_data['converged'] and not aero_data['errored_out'] and not aero_data['timed_out']:
                self.plot_mses_pressure_coefficient_distribution(aero_data, self.geo_col.container()["mea"][mea_name],
                                                                 mses_settings)
                self.switch_to_tab("Analysis")
                self.display_svgs(mset_settings, mplot_settings)

                # Update the last successful analysis directory (for easy access in field plotting)
                self.last_analysis_dir = os.path.join(mset_settings["airfoil_analysis_dir"],
                                                      mset_settings["airfoil_coord_file_name"])

                # Increment the number of converged analyses and the total number of analyses
                self.n_converged_analyses += 1
                self.n_analyses += 1
            else:
                self.n_analyses += 1
        elif status == "polar_analysis_complete" and isinstance(data, tuple):
            aero_data = data[0]
            mset_settings = data[1]
            mses_settings = data[2]
            self.display_mpolar_result(aero_data, mset_settings, mses_settings)
            self.switch_to_tab("Polars")
        elif status == "switch_to_residuals_tab":
            self.switch_to_tab("Residuals")
        elif status == "mses_residual" and isinstance(data, tuple):
            if self.residual_graph is None:
                self.residual_graph = ResidualGraph(
                    theme=self.themes[self.current_theme],
                    grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
                )
                self.add_new_tab_widget(self.residual_graph.w, "Residuals")
                self.switch_to_tab("Residuals")

            # Assign the data from the pipe to variables
            new_iteration = data[0]
            new_rms_dR = data[1]
            new_rms_dA = data[2]
            new_rms_dV = data[3]

            # If running MPOLAR and an angle of attack fails, MSES will start over at a previous iteration.
            # In this case, we need to delete the existing data from that iteration onward
            if self.residual_data and new_iteration <= self.residual_data[-1][0]:
                prev_iteration_idx = [arr[0] for arr in self.residual_data].index(new_iteration)
                del self.residual_data[prev_iteration_idx:]

            self.residual_data.append([new_iteration, new_rms_dR, new_rms_dA, new_rms_dV])
            self.residual_graph.plot_items[0].setData([arr[0] for arr in self.residual_data],
                                                      [arr[1] for arr in self.residual_data])
            self.residual_graph.plot_items[1].setData([arr[0] for arr in self.residual_data],
                                                      [arr[2] for arr in self.residual_data])
            self.residual_graph.plot_items[2].setData([arr[0] for arr in self.residual_data],
                                                      [arr[3] for arr in self.residual_data])
            self.residual_graph.set_legend_label_format(self.themes[self.current_theme])
        elif status == "clear_polar_plots":
            if self.polar_graph_collection is not None:
                self.polar_graph_collection.clear_data()
        elif status == "plot_polars" and isinstance(data, dict):
            if self.polar_graph_collection is None:
                self.polar_graph_collection = PolarGraphCollection(
                    theme=self.themes[self.current_theme],
                    grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
                )
                self.add_new_tab_widget(self.polar_graph_collection, "Polars")
                self.switch_to_tab("Polars")
            self.polar_graph_collection.set_data(data)
        elif status == "polar_progress" and isinstance(data, int):
            if self.permanent_widget.progress_bar.isHidden():
                self.permanent_widget.progress_bar.show()
            self.permanent_widget.progress_bar.setValue(data)
        elif status == "polar_complete":
            self.permanent_widget.progress_bar.hide()
        elif status == "clear_airfoil_matching_plots":
            if self.airfoil_matching_graph_collection is not None:
                self.airfoil_matching_graph_collection.clear_data()
        elif status == "symmetric_area_difference" and isinstance(data, tuple):
            current_fun_value = data[0]
            coords = data[1]
            airfoil_to_match_xy = data[2]
            self.status_bar.showMessage(f"Symmetric area difference: {current_fun_value:.3e}")
            if self.airfoil_matching_graph_collection is None:
                self.airfoil_matching_graph_collection = AirfoilMatchingGraphCollection(
                    theme=self.themes[self.current_theme],
                    grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
                )
                self.add_new_tab_widget(self.airfoil_matching_graph_collection, "Matching")
                self.switch_to_tab("Matching")
            self.airfoil_matching_graph_collection.set_data(current_fun_value, coords, airfoil_to_match_xy)
        elif status == "match_airfoil_complete":
            if not isinstance(data, OptimizeResult):
                raise ValueError(f"data ({data}) must be of type scipy.optimize.OptimizeResult")
            res = data
            msg_mode = "error"
            message = res.message
            if hasattr(res, "success") and res.success:
                update_params = res.x
                self.geo_col.assign_design_variable_values(update_params, bounds_normalized=True)
                msg_mode = "info"
                self.output_area_text(f"Airfoil matched successfully. Symmetric area difference: {res.fun:.3e}. "
                                      f"Function evaluations: {res.nfev}. Gradient evaluations: {res.njev}.",
                                      line_break=True)
                message = f"{res.message}. Geometry canvas updated with new design variable values."
            self.disp_message_box(message=message, message_mode=msg_mode)
        elif status == "resources_update":
            assert isinstance(data, tuple)
            if self.resources_graph is None:
                self.resources_graph = ResourcesGraph(
                    theme=self.themes[self.current_theme],
                    grid=self.main_icon_toolbar.buttons["grid"]["button"].isChecked()
                )
                self.add_new_tab_widget(self.resources_graph.w, "Resources")
            self.resources_graph.update(time_array=data[0], cpu_percent_array=data[1], mem_percent_array=data[2])

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

    def run_mses(self, mset_settings: dict, mses_settings: dict, mplot_settings: dict, mpolar_settings: dict):

        def remove_equations(geo_col: GeometryCollection):
            """
            Ensures that items in the geometry collection are picklable by the multiprocessing module
            """
            for param in geo_col.container()["params"].values():
                param.equation_dict = None
                param.equation = None
                param.equation_str = None
            for dv in geo_col.container()["desvar"].values():
                dv.equation_dict = None
                dv.equation = None
                dv.equation_str = None

        def run_cpu_bound_process():
            geo_col_copy = deepcopy(self.geo_col)
            remove_equations(geo_col_copy)
            mea = geo_col_copy.container()["mea"][mset_settings["mea"]]
            alfa_array = mpolar_settings["alfa_array"] if mpolar_settings is not None else None

            self.mses_process = CPUBoundProcess(
                calculate_aero_data,
                args=(
                    mset_settings["airfoil_analysis_dir"],
                    mset_settings["airfoil_coord_file_name"],
                    None,
                    mea,
                    None,
                    "MSES",
                    None,
                    mset_settings,
                    mses_settings,
                    mplot_settings,
                    mpolar_settings,
                    True,
                    True,
                    alfa_array
                )
            )
            self.mses_process.progress_emitter.signals.progress.connect(self.progress_update)
            self.mses_process.start()

        # Start running the CPU-bound process from a worker thread (separate from the main GUI thread)
        self.mses_thread = Thread(target=run_cpu_bound_process)
        self.mses_thread.start()

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

        # TODO: follow this same code architecture for XFOIL one-off analysis

    def stop_process(self):

        processes = [self.shape_opt_process, self.mses_process, self.match_airfoil_process,
                     self.display_resources_process]
        if all([process is None for process in processes]):
            # self.disp_message_box("No process to terminate")
            return

        for process in processes:
            if process is None:
                continue
            process.terminate()

        threads = [self.opt_thread, self.mses_thread, self.match_airfoil_thread, self.display_resources_thread]

        for thread in threads:
            if thread is None:
                continue
            thread.join()

        self.shape_opt_process = None
        self.mses_process = None
        self.match_airfoil_process = None
        self.display_resources_process = None

    @staticmethod
    def generate_output_folder_link_text(folder: str):
        return f"<a href='{folder}' style='font-family:DejaVu Sans Mono; " \
               f"color: #1FBBCC; font-size: 10pt;'>Open output folder</a><br>"

    def set_pool(self, pool_obj: object):
        self.pool = pool_obj

    @staticmethod
    def shape_opt_progress_callback_fn(progress_object: object):
        if isinstance(progress_object, OptCallback):
            progress_object.exec_callback()

    def shape_opt_finished_callback_fn(self, success: bool):
        self.stop_process()
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

    def load_example(self, example_name: str):
        example_file = os.path.join(EXAMPLES_DIR, example_name)
        self.load_geo_col_no_dialog(example_file)
        self.setWindowTitle(f"pymead - {os.path.split(example_file)[-1]}")

    def load_example_sc20612_match(self):
        self.load_example("match_sc20612-il.jmea")

    def load_example_basic_airfoil_sharp(self):
        self.load_example("basic_airfoil_sharp.jmea")

    def load_example_basic_airfoil_blunt(self):
        self.load_example("basic_airfoil_blunt.jmea")

    def load_example_basic_airfoil_sharp_dv(self):
        self.load_example("basic_airfoil_sharp_dv.jmea")

    def load_example_basic_airfoil_blunt_dv(self):
        self.load_example("basic_airfoil_blunt_dv.jmea")

    def load_example_isolated_propulsor(self):
        self.load_example("isolated_propulsor.jmea")

    def load_example_underwing_propulsor(self):
        self.load_example("underwing_propulsor.jmea")

    def load_example_n0012(self):
        self.load_example("n0012-il.jmea")

    def load_example_sc20612(self):
        self.load_example("sc20612-il.jmea")

    def toggle_full_screen(self):
        if not self.isMaximized():
            self.showMaximized()
        else:
            self.showNormal()

    def verify_constraints(self):
        geo_col_copy = GeometryCollection.set_from_dict_rep(self.geo_col.get_dict_rep())
        try:
            geo_col_copy.verify_all()
            self.disp_message_box("Constraint Verification Passed", message_mode="info")
        except AssertionError:
            self.disp_message_box("Constraint Verification Failed")

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key.Key_Escape:
            self.geo_col.clear_selected_objects()
            self.status_bar.clearMessage()
        if a0.key() == Qt.Key.Key_Delete:
            self.airfoil_canvas.removeSelectedObjects()
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
    sys.exit(app.exec())


if __name__ == "__main__":
    # First, we must add freeze support for multiprocessing.Pool to work properly in Windows in the version of the GUI
    # assembled by PyInstaller. This next statement affects only Windows; it has no impact on *nix OS since Pool
    # already works fine there.
    mp.freeze_support()

    # Generate the graphical user interface
    main()
