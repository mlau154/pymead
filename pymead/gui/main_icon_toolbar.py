import numpy as np
from PyQt5.QtWidgets import QToolBar, QToolButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from pymead import DATA_DIR
import os
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead.utils.read_write_files import load_data
from pymead.gui.input_dialog import SymmetryDialog
from pymead import ICON_DIR
from functools import partial


class MainIconToolbar(QToolBar):
    """Class containing the ToolBar buttons for the GUI. Note that the button settings are loaded from a JSON files
    stored in the gui directory and stored as a dict. The actual QToolButtons themselves are stored inside the
    \'buttons\' attribute."""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.new_airfoil_location = None
        self.symmetry_dialog = None
        self.icon_dir = ICON_DIR
        self.parent.addToolBar(self)
        self.button_settings = load_data("buttons.json")
        self.button_settings.pop("template")
        self.buttons = None
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

    def on_grid_button_pressed(self, checked):
        # import pyqtgraph as pg
        if checked:
            self.parent.v.showGrid(x=True, y=True)
        else:
            self.parent.v.showGrid(x=False, y=False)

    def change_background_color_button_toggled(self, checked):
        print(f"checked = {checked}")
        if checked:
            self.parent.dark_mode = True
            self.parent.set_dark_mode()
        else:
            self.parent.dark_mode = False
            self.parent.set_light_mode()

        if self.parent.analysis_graph is not None:
            if checked:
                self.parent.analysis_graph.set_background('#2a2a2b')
            else:
                self.parent.analysis_graph.set_background('w')

        if checked:
            self.parent.param_tree_instance.set_dark_mode()
        else:
            self.parent.param_tree_instance.set_light_mode()
        # QTreeView::branch:closed {color: white;} QTreeView::branch:open {color: white;}''')
        # QTreeView::item {border: 1px solid black;}''')  # need to use image, not
        # color for open closed arrows
        # self.parent.design_tree_widget.updatePalette()
        self.parent.show()

    def add_airfoil_button_toggled(self):
        print('Add airfoil button toggled!')

        def scene_clicked(ev):
            self.new_airfoil_location = self.parent.mea.v.vb.mapSceneToView(ev.scenePos())
            airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(self.new_airfoil_location.x()),
                                                                    dy=Param(self.new_airfoil_location.y())))
            self.parent.mea.te_thickness_edit_mode = self.parent.te_thickness_edit_mode
            self.parent.mea.add_airfoil(airfoil, len(self.parent.mea.airfoils), self.parent.param_tree_instance)
            self.parent.airfoil_name_list = [k for k in self.parent.mea.airfoils.keys()]
            self.parent.param_tree_instance.p.child("Analysis").child("Inviscid Cl Calc").setLimits([a.tag for a in self.parent.mea.airfoils.values()])
            self.parent.param_tree_instance.params[-1].add_airfoil(airfoil, len(self.parent.mea.airfoils) - 1)
            self.parent.mea.v.scene().sigMouseClicked.disconnect()
            airfoil.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.parent.param_tree_instance.plot_changed,
                                                                         f"A{len(self.parent.mea.airfoils) - 1}"))

        self.parent.mea.v.scene().sigMouseClicked.connect(scene_clicked)

    def te_thickness_mode_toggled(self, checked):
        if checked:
            for airfoil in self.parent.mea.airfoils.values():
                airfoil.airfoil_graph.te_thickness_edit_mode = True
            self.parent.te_thickness_edit_mode = True
        else:
            for airfoil in self.parent.mea.airfoils.values():
                airfoil.airfoil_graph.te_thickness_edit_mode = False
            self.parent.te_thickness_edit_mode = False

    @pyqtSlot(str)
    def symmetry_connection(self, obj: str):
        if self.symmetry_dialog:
            self.symmetry_dialog.inputs[self.symmetry_dialog.current_form_idx][1].setText(obj)

    def on_symmetry_button_pressed(self):
        self.parent.param_tree_instance.t.sigSymmetry.connect(self.symmetry_connection)
        self.symmetry_dialog = SymmetryDialog(self)
        self.symmetry_dialog.show()
        self.symmetry_dialog.accepted.connect(self.make_symmetric)

    def make_symmetric(self):
        outputs = self.symmetry_dialog.getInputs()
        print(f"outputs = {outputs}")
