from PyQt5.QtWidgets import QToolBar, QToolButton, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import os
import pyqtgraph as pg
from pymead.utils.read_write_files import load_data
# from pymead.core.symmetry import symmetry
from pymead import ICON_DIR, GUI_SETTINGS_DIR, q_settings
# from functools import partial


class MainIconToolbar(QToolBar):
    """Class containing the ToolBar buttons for the GUI. Note that the button settings are loaded from a JSON files
    stored in the gui directory and stored as a dict. The actual QToolButtons themselves are stored inside the
    \'buttons\' attribute."""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.new_airfoil_location = None
        self.symmetry_dialog = None
        self.pos_constraint_dialog = None
        self.icon_dir = ICON_DIR
        self.parent.addToolBar(self)
        self.button_settings = load_data(os.path.join(GUI_SETTINGS_DIR, "buttons.json"))
        self.button_settings.pop("template")
        self.buttons = None
        self.browser = None
        self.add_all_buttons()

    def add_all_buttons(self):
        self.buttons = {}
        for button_name, button_settings in self.button_settings.items():
            self.buttons[button_name] = {}
            # Add the icon:
            self.buttons[button_name]["icon"] = QIcon(os.path.join(self.icon_dir, button_settings["icon"]))

            # Set up the physical button:
            button = QToolButton(self)
            self.buttons[button_name]["button"] = button
            button.setStatusTip(button_settings["status_tip"])  # add the status tip
            button.setCheckable(button_settings["checkable"])  # determine whether the button can be checked
            button.setIcon(self.buttons[button_name]["icon"])  # set the icon for the button
            if button_settings["checkable"]:
                button.toggled.connect(getattr(self, button_settings["function"]))  # add the functionality
            else:
                button.clicked.connect(getattr(self, button_settings["function"]))  # add the functionality
            if button.isCheckable():
                if button_settings["checked-by-default"]:
                    button.toggle()
            self.addWidget(button)

    def on_stop_button_pressed(self):
        self.parent.stop_optimization()

    def on_grid_button_pressed(self):
        self.parent.airfoil_canvas.toggleGrid()
        for dw in self.parent.dock_widgets:
            if isinstance(dw.widget(), pg.GraphicsLayoutWidget):
                v = dw.widget().getItem(0, 0)
                x_state = v.ctrl.xGridCheck.checkState()
                y_state = v.ctrl.yGridCheck.checkState()
                if x_state or y_state:
                    v.showGrid(x=False, y=False)
                else:
                    v.showGrid(x=True, y=True)

    def change_background_color_button_toggled(self, checked):

        if checked:
            self.parent.set_theme("dark")
            q_settings.setValue("dark_theme_checked", 2)
        else:
            self.parent.set_theme("light")
            q_settings.setValue("dark_theme_checked", 0)

        self.parent.show()

    def on_draw_points_pressed(self):
        self.parent.airfoil_canvas.drawPoints()

    def on_draw_lines_pressed(self):
        self.parent.airfoil_canvas.drawLines()

    def on_draw_bezier_pressed(self):
        self.parent.airfoil_canvas.drawBeziers()

    def on_add_distance_constraint_pressed(self):
        self.parent.airfoil_canvas.addDistanceConstraint()

    def on_add_rel_angle_constraint_pressed(self):
        self.parent.airfoil_canvas.addRelAngle3Constraint()

    def on_add_perp_constraint_pressed(self):
        self.parent.airfoil_canvas.addPerp3Constraint()

    def on_add_anti_parallel_constraint_pressed(self):
        self.parent.airfoil_canvas.addAntiParallel3Constraint()

    def on_add_symmetry_constraint_pressed(self):
        self.parent.airfoil_canvas.addSymmetryConstraint()

    def on_add_roc_constraint_pressed(self):
        self.parent.airfoil_canvas.addROCurvatureConstraint()

    def on_help_button_pressed(self):
        self.parent.show_help()
