from PyQt5.QtWidgets import QToolBar, QToolButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import os
import pyqtgraph as pg
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.utils.read_write_files import load_data
from pymead.gui.input_dialog import SymmetryDialog, PosConstraintDialog
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
        # import pyqtgraph as pg
        if hasattr(self.parent, "current_dock_widget"):
            dw = self.parent.current_dock_widget
        else:
            dw = self.parent.first_dock_widget
        if dw is None:
            self.parent.airfoil_canvas.toggleGrid()
        else:
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

    def add_airfoil_button_toggled(self):

        def scene_clicked(ev):
            self.new_airfoil_location = self.parent.v.vb.mapSceneToView(ev.scenePos())
            self.parent.v.scene().sigMouseClicked.disconnect()
            airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(self.new_airfoil_location.x()),
                                                                    dy=Param(self.new_airfoil_location.y())))
            self.parent.add_airfoil(airfoil)

        self.parent.v.scene().sigMouseClicked.connect(scene_clicked)

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
                'xy': f"x1={out['x1']}, y1={out['y1']}, theta_rad={out['angle']}, xy={out['tool']}.xy, "
                      f"alf_tool={tool_base}.alf, alf_target={target_base}.alf, c_tool={tool_base}.c,"
                      f" c_target={target_base}.c, dx_tool={tool_base}.dx, dx_target={target_base}.dx, "
                      f"dy_tool={tool_base}.dy, dy_target={target_base}.dy",
                'phi': f"x1={out['x1']}, y1={out['y1']}, theta_rad={out['angle']}, alf_tool={tool_base}.alf, "
                       f"alf_target={target_base}.alf, phi={out['tool']}.phi, upper_target={upper_target}, "
                       f"upper_tool={upper_tool}",
                'psi1': f"psi1={out['tool']}.psi1",
                'psi2': f"psi2={out['tool']}.psi2",
                'r': f"r={out['tool']}.r",
                'L': f"L={out['tool']}.L, c_target={target_base}.c, c_tool={tool_base}.c",
                'R': f"R={out['tool']}.R, c_target={target_base}.c, c_tool={tool_base}.c",
            }
            eq_string = f"^symmetry.symmetry(name, {extra_args[param_name]})"
            self.parent.param_tree_instance.block_changes(eq)
            eq.setValue(eq_string)
            self.parent.param_tree_instance.flush_changes(eq)
            self.parent.param_tree_instance.update_equation(eq, eq_string)
            if isinstance(param.airfoil_param, PosParam):
                param.airfoil_param.linked = (True, True)
            else:
                param.airfoil_param.linked = True

        airfoil_param_tree = self.parent.param_tree_instance.p.child('Airfoil Parameters')
        out = self.symmetry_dialog.getInputs()
        target = out['target'].replace('$', '')
        target_list = target.split('.')
        tool = out['tool'].replace('$', '')
        tool_list = tool.split('.')
        tool_base = f"${tool_list[0]}.Base"
        target_base = f"${target_list[0]}.Base"
        if 'FreePoints' in target_list:
            fp_or_ap = 'fp'
            assign_equation('xy')
        elif 'AnchorPoints' in target_list:
            fp_or_ap = 'ap'
            for param_str in ['xy', 'phi', 'psi1', 'psi2', 'L', 'r', 'R']:
                assign_equation(param_str)
        else:
            raise ValueError('Target selection must be either a FreePoint or an AnchorPoint')

    @pyqtSlot(str)
    def pos_constraint_connection(self, obj: str):
        if self.pos_constraint_dialog:
            self.pos_constraint_dialog.inputs[self.pos_constraint_dialog.current_form_idx][1].setText(obj)

    def on_pos_constraint_pressed(self):
        self.parent.param_tree_instance.t.sigPosConstraint.connect(
            self.pos_constraint_connection)  # connect parameter selection to the QLineEdits in the dialog
        self.pos_constraint_dialog = PosConstraintDialog(self)  # generate the dialog
        self.pos_constraint_dialog.show()  # show the dialog
        self.pos_constraint_dialog.accepted.connect(self.constrain_position)  # apply symmetry equations once OK is pressed

    def constrain_position(self):

        def get_grandchild(param_tree, child_list: list, param_name: str = None):
            # print(f"{target_list = }")
            current_param = param_tree.child(target_list[0])
            # print(f"{current_param = }")
            for idx in range(1, len(child_list)):
                current_param = current_param.child(child_list[idx])
            # print(f"Now, {current_param = }")
            if param_name is not None:
                full_param_name = f"{'.'.join(child_list)}.{param_name}"
                return current_param.child(full_param_name)
            else:
                return current_param

        airfoil_param_tree = self.parent.param_tree_instance.p.child('Airfoil Parameters')
        out = self.pos_constraint_dialog.getInputs()
        target = out['target'].replace('$', '')
        target_list = target.split('.')
        tool = out['tool'].replace('$', '')
        tool_list = tool.split('.')
        if 'Custom' in target_list:
            param = get_grandchild(airfoil_param_tree, target_list)
        else:
            param = get_grandchild(airfoil_param_tree, target_list, 'xy')
        if 'Custom' not in tool_list:
            out['tool'] = out['tool'] + '.xy'
        self.parent.param_tree_instance.add_equation_box(param)
        eq = param.child('Equation Definition')
        if 'dx' in out.keys():
            eq_string = "{%s[0] + %s, %s[1] + %s}" % (out['tool'], out['dx'], out['tool'], out['dy'])
        else:
            eq_string = "{%s[0] + %s * cos(%s), %s[1] + %s * sin(%s)}" % (out['tool'], out['dist'],
                                                                          out['angle'], out['tool'],
                                                                          out['dist'], out['angle'])
        self.parent.param_tree_instance.block_changes(eq)
        eq.setValue(eq_string)
        self.parent.param_tree_instance.flush_changes(eq)
        self.parent.param_tree_instance.update_equation(eq, eq_string)

    def on_help_button_pressed(self):
        self.parent.show_help()
