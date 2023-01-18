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
from pymead.core.symmetry import symmetry
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
        self.parent.param_tree_instance.t.sigSymmetry.connect(self.symmetry_connection)  # connect parameter selection
        # to the QLineEdits in the dialog
        self.symmetry_dialog = SymmetryDialog(self)  # generate the dialog
        self.symmetry_dialog.show()  # show the dialog
        self.symmetry_dialog.accepted.connect(self.make_symmetric)  # apply symmetry equations once OK is pressed

    def make_symmetric(self):
        def get_grandchild(param_tree, child_list: list, param_name: str):
            current_param = param_tree.child(target_list[0])
            for idx in range(1, len(child_list)):
                current_param = current_param.child(child_list[idx])
            full_param_name = f"{'.'.join(child_list)}.{param_name}"
            return current_param.child(full_param_name)

        def assign_equation(param_name: str):
            param = get_grandchild(airfoil_param_tree, target_list, param_name)
            self.parent.param_tree_instance.add_equation_box(param)
            eq = param.child('Equation Definition')
            upper_target, upper_tool = False, False
            if fp_or_ap == 'ap':
                ap_order_target = self.parent.mea.airfoils[target_list[0]].anchor_point_order
                ap_order_tool = self.parent.mea.airfoils[tool_list[0]].anchor_point_order
                ap_target = target_list[2]  # e.g., 'ap0'
                ap_tool = tool_list[2]  # e.g., 'ap1'
                if ap_order_target.index(ap_target) < ap_order_target.index('le'):
                    upper_target = True
                if ap_order_tool.index(ap_tool) < ap_order_tool.index('le'):
                    upper_tool = True
            extra_args = {
                'xp': f"x1={out['x1']}, y1={out['y1']}, theta_rad={out['angle']}, xp={out['tool']}.xp, "
                      f"yp={out['tool']}.yp",
                'yp': f"x1={out['x1']}, y1={out['y1']}, theta_rad={out['angle']}, xp={out['tool']}.xp, "
                      f"yp={out['tool']}.yp",
                'phi': f"x1={out['x1']}, y1={out['y1']}, theta_rad={out['angle']}, alf_tool=${tool_list[0]}.Base.alf, "
                       f"alf_target=${target_list[0]}.Base.alf, phi={out['tool']}.phi, upper_target={upper_target}, "
                       f"upper_tool={upper_tool}",
                'psi1': f"psi1={out['tool']}.psi1",
                'psi2': f"psi2={out['tool']}.psi2",
                'r': f"r={out['tool']}.r",
                'L': f"L={out['tool']}.L",
                'R': f"R={out['tool']}.R",
            }
            eq_string = f"symmetry(param_name, {extra_args[param_name]})"
            self.parent.param_tree_instance.block_changes(eq)
            eq.setValue(eq_string)
            self.parent.param_tree_instance.flush_changes(eq)
            self.parent.param_tree_instance.update_equation(eq, eq_string, symmetry=symmetry, param_name=param_name)

        airfoil_param_tree = self.parent.param_tree_instance.p.child('Airfoil Parameters')
        out = self.symmetry_dialog.getInputs()
        target = out['target'].replace('$', '')
        target_list = target.split('.')
        tool = out['tool'].replace('$', '')
        tool_list = tool.split('.')
        if 'FreePoints' in target_list:
            fp_or_ap = 'fp'
            for param_str in ['xp', 'yp']:
                assign_equation(param_str)
        elif 'AnchorPoints' in target_list:
            fp_or_ap = 'ap'
            for param_str in ['xp', 'yp', 'phi', 'psi1', 'psi2', 'L', 'r', 'R']:
                assign_equation(param_str)
        else:
            raise ValueError('Target selection must be either a FreePoint or an AnchorPoint')
