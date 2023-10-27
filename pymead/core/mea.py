from copy import deepcopy
import importlib.util
import itertools
import numpy as np
import os
import typing

import benedict

from pymead.core.airfoil import Airfoil
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.utils.dict_recursion import set_all_dict_values, assign_airfoil_tags_to_param_dict, \
    assign_names_to_params_in_param_dict, unravel_param_dict_deepcopy
from pymead.utils.read_write_files import save_data
from pymead.plugins.IGES.iges_generator import IGESGenerator
from pymead.plugins.IGES.curves import BezierIGES
from pymead import DATA_DIR, PLUGINS_DIR, INCLUDE_FILES


class MEA:
    """
    Class for multi-element airfoils. Serves as a container for ``pymead.core.airfoil.Airfoil``s and adds a few methods
    important for the Graphical User Interface.
    """
    def __init__(self, param_tree=None, airfoils: Airfoil or typing.List[Airfoil, ...] or None = None,
                 airfoil_graphs_active: bool = False):
        self.airfoils = {}
        self.file_name = None
        self.user_mods = None
        self.param_tree = param_tree
        self.param_dict = {'Custom': {}}
        self.airfoil_graphs_active = airfoil_graphs_active
        self.te_thickness_edit_mode = False
        self.w = None
        self.v = None
        if not isinstance(airfoils, list):
            if airfoils is not None:
                self.add_airfoil(airfoils, 0, param_tree)
        else:
            for idx, airfoil in enumerate(airfoils):
                self.add_airfoil(airfoil, idx, param_tree)

    # def __getstate__(self):
    #     """
    #     Reimplemented to ensure MEA picklability
    #     """
    #     state = self.__dict__.copy()
    #     if state['v'] is not None:
    #         state['v'].clear()
    #     state['w'] = None  # Set unpicklable GraphicsLayoutWidget object to None
    #     state['v'] = None  # Set unpicklable ViewBox object to None
    #     return state

    def add_airfoil(self, airfoil: Airfoil, idx: int, param_tree, w=None, v=None, gui_obj=None):
        """
        Add an airfoil at index ``idx`` to the multi-element airfoil container.

        Parameters
        ==========
        airfoil: Airfoil
            Airfoil to insert into the MEA container

        idx: int
            Insertion index (0 corresponds to insertion at the beginning of the list)

        param_tree
            Parameter Tree from the GUI which is added as an airfoil attribute following insertion

        w
            A ``pyqtgraph`` ``GraphicsLayoutWidget`` associated with the GUI. Used to link the airfoil to its graph
            in the GUI.

        v
            A ``pyqtgraph`` ``PlotDataItem`` associated with the GUI. Used to link the airfoil to its graph in the GUI.
        """
        if airfoil.tag is None:
            airfoil.tag = f'A{idx}'
        # print(f"Adding mea to airfoil {airfoil.tag}")
        airfoil.mea = self
        self.airfoils[airfoil.tag] = airfoil
        self.param_dict[airfoil.tag] = airfoil.param_dicts
        # print(f"param_dict = {self.param_dict}")

        set_all_dict_values(self.param_dict[airfoil.tag])

        assign_airfoil_tags_to_param_dict(self.param_dict[airfoil.tag], airfoil_tag=airfoil.tag)

        assign_names_to_params_in_param_dict(self.param_dict)

        if self.airfoil_graphs_active:
            self.add_airfoil_graph_to_airfoil(airfoil, idx, param_tree, w=w, v=v, gui_obj=gui_obj)

        dben = benedict.benedict(self.param_dict)
        for k in dben.keypaths():
            param = dben[k]
            if isinstance(param, Param):
                if param.mea is None:
                    param.mea = self
                if param.mea.param_tree is None:
                    param.mea.param_tree = self.param_tree

    def remove_airfoil(self, airfoil_tag: str):
        # Remove all items from the AirfoilGraph corresponding to this airfoil
        airfoil_graph = self.airfoils[airfoil_tag].airfoil_graph
        if airfoil_graph is not None:
            for curve in self.airfoils[airfoil_tag].curve_list[::-1]:
                airfoil_graph.v.removeItem(curve.pg_curve_handle)
            airfoil_graph.v.removeItem(airfoil_graph.polygon_item)
            airfoil_graph.v.removeItem(airfoil_graph)

        # Remove the airfoil from the ParameterTree
        if self.param_tree is not None:
            self.param_tree.p.child("Airfoil Parameters").child(airfoil_tag).remove()
            # self.param_tree.airfoil_headers.pop(airfoil_tag)

        # Remove the airfoil from the MEA
        self.param_dict.pop(airfoil_tag)
        self.airfoils.pop(airfoil_tag)

    def assign_names_to_params_in_param_dict(self):
        """
        Recursively assigns the name of the airfoil along with all its base parameters, ``FreePoint``s, and
        ``AnchorPoints`` to the parameter dictionary.
        """
        assign_names_to_params_in_param_dict(self.param_dict)

    def add_airfoil_graph_to_airfoil(self, airfoil: Airfoil, idx: int, param_tree, w=None, v=None, gui_obj=None):
        """
        Add a ``pyqtgraph``-based ``pymead.gui.airfoil_graph.AirfoilGraph`` to the airfoil at index ``int``.
        """
        from pymead.gui.airfoil_graph import AirfoilGraph
        if w is None:
            if idx == 0:
                airfoil_graph = AirfoilGraph(airfoil)
                self.w = airfoil_graph.w
                self.v = airfoil_graph.v
                # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
                airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
            else:  # Assign the first airfoil's Graphics Window and ViewBox to each subsequent airfoil
                airfoil_graph = AirfoilGraph(airfoil,
                                             w=self.airfoils['A0'].airfoil_graph.w,
                                             v=self.airfoils['A0'].airfoil_graph.v, gui_obj=gui_obj)
                # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
                airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
        else:
            # print("Creating new AirfoilGraph!")
            airfoil_graph = AirfoilGraph(airfoil, w=w, v=v, gui_obj=gui_obj)
            # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
            airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
            self.w = w
            self.v = v

        airfoil_graph.param_tree = param_tree
        airfoil.airfoil_graph = airfoil_graph

    def extract_parameters(self):
        """
        Extracts the 1-D list of parameters from the airfoil system corresponding to all the parameters with
        ``active=True`` and ``linked=False``. Any ``PosParam`` corresponds to two consecutive parameters. All
        parameters are normalized between their respective lower bounds (``bounds[0]``) and upper bounds (``bounds[1]``)
        such that all values are between 0 and 1.

        Returns
        =======
        list
            1-D list of normalized parameter values
        """

        parameter_list = []
        norm_value_list = []

        def check_for_bounds_recursively(d: dict, bounds_error_=False):
            for k_, v in d.items():
                if not bounds_error_:
                    if isinstance(v, dict):
                        bounds_error_, param_name_ = check_for_bounds_recursively(v, bounds_error_)
                    else:
                        if isinstance(v, Param) and not isinstance(v, PosParam):
                            if v.active and not v.linked:
                                if v.bounds[0] == -np.inf or v.bounds[0] == np.inf or v.bounds[1] == -np.inf or v.bounds[1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                        elif isinstance(v, PosParam):
                            if v.active[0] and not v.linked[0]:
                                if v.bounds[0][0] == -np.inf or v.bounds[0][0] == np.inf or v.bounds[0][1] == -np.inf or v.bounds[0][1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                                if v.bounds[1][0] == -np.inf or v.bounds[1][0] == np.inf or v.bounds[1][1] == -np.inf or v.bounds[1][1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                        else:
                            raise ValueError('Found value in dictionary not of type \'Param\' or \'PosParam\'')
                else:
                    return bounds_error_, None
            return bounds_error_, None

        def extract_parameters_recursively(d: dict):
            for k_, v in d.items():
                if isinstance(v, dict):
                    extract_parameters_recursively(v)
                else:
                    if isinstance(v, Param) and not isinstance(v, PosParam):
                        if v.active and not v.linked:
                            norm_value_list.append((v.value - v.bounds[0]) / (v.bounds[1] - v.bounds[0]))
                            parameter_list.append(v)
                    elif isinstance(v, PosParam):
                        if v.active[0] and not v.linked[0]:
                            norm_value_list.append((v.value[0] - v.bounds[0][0]) / (v.bounds[0][1] - v.bounds[0][0]))
                            parameter_list.append(v)
                        if v.active[1] and not v.linked[1]:
                            norm_value_list.append((v.value[1] - v.bounds[1][0]) / (v.bounds[1][1] - v.bounds[1][0]))
                            if v not in parameter_list:  # only add the Parameter if it hasn't already been added
                                parameter_list.append(v)
                    else:
                        raise ValueError('Found value in dictionary not of type \'Param\' or \'PosParam\'')

        bounds_error, param_name = check_for_bounds_recursively(self.param_dict)
        if bounds_error:
            error_message = f'Bounds must be set for each active and unlinked parameter for parameter extraction (at ' \
                            f'least one infinite bound found for {param_name})'
            print(error_message)
            return error_message, None
        else:
            extract_parameters_recursively(self.param_dict)
            # parameter_list = np.loadtxt(os.path.join(DATA_DIR, 'parameter_list.dat'))
            # self.update_parameters(parameter_list)
            # fig_.savefig(os.path.join(DATA_DIR, 'test_airfoil.png'))
            return norm_value_list, parameter_list

    def copy_as_param_dict(self, deactivate_airfoil_graphs: bool = False):
        """
        Copies the entire airfoil system as a Python dictionary. This dictionary can later be converted back to an
        airfoil system using the class method ``generate_from_param_dict``.

        Parameters
        ==========
        deactivate_airfoil_graphs: bool
            This argument should be set to ``True`` if the target case for re-loading the airfoil system is in a script
            rather than the GUI. Default: ``False``.

        Returns
        =======
        dict
            A Python dictionary describing the airfoil system (corresponds to the ``.jmea`` files in the GUI).
        """
        output_dict_ = {}
        unravel_param_dict_deepcopy(self.param_dict, output_dict=output_dict_)
        for k, v in output_dict_.items():
            if k != 'Custom':
                output_dict_[k]['anchor_point_order'] = deepcopy(self.airfoils[k].anchor_point_order)
                output_dict_[k]['free_point_order'] = deepcopy(self.airfoils[k].free_point_order)
        output_dict_['file_name'] = self.file_name
        if deactivate_airfoil_graphs:
            output_dict_['airfoil_graphs_active'] = False
        else:
            output_dict_['airfoil_graphs_active'] = self.airfoil_graphs_active
        return deepcopy(output_dict_)

    def save_airfoil_system(self, file_name: str):
        """
        Saves the encapsulated airfoil system as a ``.jmea`` file.

        Parameters
        ==========
        file_name: str
          The file location where the MEA is to be stored (relative or absolute path). If the file name does not
          end with ".jmea", the ``.jmea`` file extension will be added automatically.
        """
        if os.path.splitext(file_name)[-1] != '.jmea':
            file_name += '.jmea'
        self.file_name = file_name
        mea_dict = self.copy_as_param_dict(deactivate_airfoil_graphs=True)
        save_data(mea_dict, file_name)

    def deepcopy(self, deactivate_airfoil_graphs: bool = False):
        """
        Composite function combining, in order, the ``copy_as_param_dict`` and ``generate_from_param_dict`` methods.
        Copying the airfoil system this way avoids the complications associated with using the standard ``deepcopy``
        function on some stored functions and ``Qt`` objects.

        Parameters
        ==========
        deactivate_airfoil_graphs: bool
            Deactivates the airfoil graph objects in the GUI. Default: ``False``.

        Returns
        =======
        MEA
            Airfoil system object
        """
        return MEA.generate_from_param_dict(self.copy_as_param_dict(deactivate_airfoil_graphs))

    def write_to_IGES(self, file_name: str):
        """
        Writes the airfoil system to file using the IGES file format.

        Parameters
        ==========
        file_name: str
            Path to IGES file
        """
        bez_IGES_entities = [
            [BezierIGES(np.column_stack((c.P[:, 0], np.zeros(len(c.P)), c.P[:, 1]))) for c in a.curve_list]
            for a in self.airfoils.values()]
        entities_flattened = list(itertools.chain.from_iterable(bez_IGES_entities))
        iges_generator = IGESGenerator(entities_flattened)
        iges_generator.generate(file_name)

    def update_parameters(self, norm_value_list: list or np.ndarray):
        """
        Updates the airfoil system using the set of normalized parameter values extracted using ``extract_parameters``.

        Parameters
        ==========
        norm_value_list: list or np.ndarray
            List of normalized parameter values from ``extract_parameters``
        """

        if isinstance(norm_value_list, list):
            norm_value_list = np.array(norm_value_list)

        if norm_value_list.ndim == 0:
            norm_value_list = np.array([norm_value_list])

        def check_for_bounds_recursively(d: dict, bounds_error_=False):
            for k, v in d.items():
                if not bounds_error_:
                    if isinstance(v, dict):
                        bounds_error_, param_name_ = check_for_bounds_recursively(v, bounds_error_)
                    else:
                        if isinstance(v, Param) and not isinstance(v, PosParam):
                            if v.active and not v.linked:
                                if v.bounds[0] == -np.inf or v.bounds[0] == np.inf or v.bounds[1] == -np.inf or \
                                        v.bounds[1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                        elif isinstance(v, PosParam):
                            if v.active[0] and not v.linked[0]:
                                if v.bounds[0][0] == -np.inf or v.bounds[0][0] == np.inf or v.bounds[0][1] == -np.inf or v.bounds[0][1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                            if v.active[1] and not v.linked[1]:
                                if v.bounds[1][0] == -np.inf or v.bounds[1][0] == np.inf or v.bounds[1][1] == -np.inf or v.bounds[1][1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                        else:
                            raise ValueError('Found value in dictionary not of type \'Param\'')
                else:
                    return bounds_error_, None
            return bounds_error_, None

        def update_parameters_recursively(d: dict, list_counter: int):
            for k, v in d.items():
                if isinstance(v, dict):
                    list_counter = update_parameters_recursively(v, list_counter)
                else:
                    if isinstance(v, Param) and not isinstance(v, PosParam):
                        if v.active and not v.linked:
                            v.value = norm_value_list[list_counter] * (v.bounds[1] - v.bounds[0]) + v.bounds[0]
                            # v.update()
                            list_counter += 1
                    elif isinstance(v, PosParam):
                        temp_xy_value = v.value  # set up a temp variable because we want to update x and y simultaneously
                        if v.active[0] and not v.linked[0]:
                            temp_xy_value[0] = norm_value_list[list_counter] * (v.bounds[0][1] - v.bounds[0][0]) + v.bounds[0][0]
                            list_counter += 1
                        if v.active[1] and not v.linked[1]:
                            temp_xy_value[1] = norm_value_list[list_counter] * (v.bounds[1][1] - v.bounds[1][0]) + v.bounds[1][0]
                            list_counter += 1
                        v.value = temp_xy_value  # replace the PosParam value with the temp value (unchanged if
                        # neither x nor y are active)
                        # print(f"Updated PosParam value! {v.value = }")
                    else:
                        raise ValueError('Found value in dictionary not of type \'Param\'')
            return list_counter

        bounds_error, param_name = check_for_bounds_recursively(self.param_dict)
        if bounds_error:
            error_message = f'Bounds must be set for each active and unlinked parameter for parameter update ' \
                            f'(at least one infinite bound found for {param_name})'
            print(error_message)
            return error_message
        else:
            for _ in range(2):  # Do this code twice to ensure everything is updated properly:
                update_parameters_recursively(self.param_dict, 0)

                for a_tag, airfoil in self.airfoils.items():
                    airfoil.update()
                    if self.airfoil_graphs_active:
                        airfoil.airfoil_graph.data['pos'] = airfoil.control_point_array
                        airfoil.airfoil_graph.updateGraph()
                        airfoil.airfoil_graph.plot_change_recursive(
                            airfoil.airfoil_graph.airfoil_parameters.child(a_tag).children())

    # def deactivate_airfoil_matching_params(self, target_airfoil: str):
    #     def deactivate_recursively(d: dict):
    #         for k, v in d.items():
    #             if isinstance(v, dict):
    #                 # If the dictionary key is equal to the target_airfoil_str, stop the recursion for this branch,
    #                 # otherwise continue with the recursion:
    #                 if k != target_airfoil:
    #                     deactivate_recursively(v)
    #             elif isinstance(v, Param) and not isinstance(v, PosParam):
    #                 if v.active:
    #                     v.active = False
    #                     v.deactivated_for_airfoil_matching = True
    #             elif isinstance(v, PosParam):
    #                 if v.active[0]:
    #                     v.active[0] = False
    #                     v.deactivated_for_airfoil_matching = True
    #                 if v.active[1]:
    #                     v.active[1] = False
    #                     v.deactivated_for_airfoil_matching = True
    #             else:
    #                 raise ValueError('Found value in dictionary not of type \'Param\'')
    #
    #     deactivate_recursively(self.param_dict)
    #     deactivate_target_params = ['dx', 'dy', 'alf', 'c', 't_te', 'r_te', 'phi_te']
    #     for p_str in deactivate_target_params:
    #         p = self.airfoils[target_airfoil].param_dicts['Base'][p_str]
    #         if p.active:
    #             p.active = False
    #             p.deactivated_for_airfoil_matching = True
    #
    # def activate_airfoil_matching_params(self, target_airfoil: str):
    #     def activate_recursively(d: dict):
    #         for k, v in d.items():
    #             if isinstance(v, dict):
    #                 # If the dictionary key is equal to the target_airfoil_str, stop the recursion for this branch,
    #                 # otherwise continue with the recursion:
    #                 if k != target_airfoil:
    #                     activate_recursively(v)
    #             elif isinstance(v, Param):
    #                 if v.deactivated_for_airfoil_matching:
    #                     v.active = True
    #                     v.deactivated_for_airfoil_matching = False
    #             else:
    #                 raise ValueError('Found value in dictionary not of type \'Param\'')
    #
    #     activate_recursively(self.param_dict)
    #     activate_target_params = ['dx', 'dy', 'alf', 'c']
    #     for p_str in activate_target_params:
    #         p = self.airfoils[target_airfoil].param_dicts['Base'][p_str]
    #         if p.deactivated_for_airfoil_matching:
    #             p.active = True
    #             p.deactivated_for_airfoil_matching = False

    def calculate_max_x_extent(self):
        """
        Calculates the maximum :math:`x`-value of the airfoil system using the absolute location of the trailing edge
        of each airfoil.

        Returns
        =======
        float
            Left-most trailing edge position of all airfoils in the airfoil system
        """
        x = None
        for a in self.airfoils.values():
            x_max = a.c.value * np.cos(a.alf.value) + a.dx.value
            if x is None:
                x = x_max
            else:
                if x_max > x:
                    x = x_max
        return x

    def add_custom_parameters(self, params: dict):
        if 'Custom' not in self.param_dict.keys():
            self.param_dict['Custom'] = {}
        for k, v in params.items():
            if hasattr(v['value'], '__iter__'):
                self.param_dict['Custom'][k] = PosParam(**v)
            else:
                self.param_dict['Custom'][k] = Param(**v)
            self.param_dict['Custom'][k].param_dict = self.param_dict
            self.param_dict['Custom'][k].mea = self

    def get_keys(self):
        """
        Used in ``pymead.core.parameter_tree.MEAParamTree`` to update the equation variable AutoCompleter
        """
        d_ben = benedict.benedict(self.param_dict)
        keypaths = d_ben.keypaths()
        elems_to_remove = []
        for idx, elem in enumerate(keypaths):
            split = elem.split('.')  # separate the keys into lists at the periods
            if len(split) < 3 and not split[0] == 'Custom':
                elems_to_remove.append(idx)
            if len(split) > 1:
                if len(split) < 4 and split[1] in ['FreePoints', 'AnchorPoints'] and idx not in elems_to_remove:
                    elems_to_remove.append(idx)

        for rem in elems_to_remove[::-1]:
            keypaths.pop(rem)

        for idx, el in enumerate(keypaths):
            keypaths[idx] = '$' + el

        return keypaths

    def get_curve_bounds(self):
        """
        Calculates the :math:`x`- and :math:`y`-ranges corresponding to the rectangle which just encapsulates the entire
        airfoil system.

        Returns
        =======
        typing.Tuple[tuple]
            :math:`x`- and :math:`y`-ranges in the format ``(x_min,x_max), (y_min,y_max)``
        """
        x_range, y_range = (None, None), (None, None)
        for a in self.airfoils.values():
            if not a.airfoil_graph:
                raise ValueError('pyqtgraph curves must be initialized to get curve bounds')
            for c in a.curve_list:
                curve = c.pg_curve_handle
                x_lims = curve.dataBounds(ax=0)
                y_lims = curve.dataBounds(ax=1)
                if x_range[0] is None:
                    x_range = x_lims
                    y_range = y_lims
                else:
                    if x_lims[0] < x_range[0]:
                        x_range = (x_lims[0], x_range[1])
                    if x_lims[1] > x_range[1]:
                        x_range = (x_range[0], x_lims[1])
                    if y_lims[0] < y_range[0]:
                        y_range = (y_lims[0], y_range[1])
                    if y_lims[1] > y_range[1]:
                        y_range = (y_range[0], y_lims[1])

                # Check if control points are out of range
                x_ctrlpt_range = (np.min(c.P[:, 0]), np.max(c.P[:, 0]))
                y_ctrlpt_range = (np.min(c.P[:, 1]), np.max(c.P[:, 1]))
                if x_ctrlpt_range[0] < x_range[0]:
                    x_range = (x_ctrlpt_range[0], x_range[1])
                if x_ctrlpt_range[1] > x_range[1]:
                    x_range = (x_range[0], x_ctrlpt_range[1])
                if y_ctrlpt_range[0] < y_range[0]:
                    y_range = (y_ctrlpt_range[0], y_range[1])
                if y_ctrlpt_range[1] > y_range[1]:
                    y_range = (y_range[0], y_ctrlpt_range[1])
        return x_range, y_range

    def get_ctrlpt_dict(self, zero_col: int = 1):
        """
        Gets the set of ControlPoints for each airfoil curve and arranges them in a format appropriate for the
        JSON file format. The keys at the top level of the dict represent the airfoil name, and each value contains a
        3-D list. The slices of the list represent the Bézier curve, the rows represent the ControlPoints, and the
        columns represent :math:`x`, :math:`y`, and :math:`z`.

        Parameters
        ==========
        zero_col: int
          The column into which the row of zeros should be placed to map the 2-D airfoil control points into 3-D space.
          For example, inserting into the first column means the airfoil will be located in the X-Z plane. Valid values:
          0, 1, or 2. Default: 1.

        Returns
        =======
        dict
          The dictionary containing the ControlPoints.
        """
        ctrlpt_dict = {}
        for a_name, a in self.airfoils.items():
            a.update()
            ctrlpts = []
            for c in a.curve_list:
                P = deepcopy(c.P)
                P = np.insert(P, zero_col, 0.0, axis=1)
                ctrlpts.append(P)
            ctrlpt_dict[a_name] = ctrlpts
        return ctrlpt_dict

    def write_NX_macro(self, fname: str, opts: dict):
        with open(fname, 'w') as f:
            for import_ in ['math', 'NXOpen', 'NXOpen.Features', 'NXOpen.GeometricUtilities', 'time']:
                f.write(f'import {import_}\n')

            with open(os.path.join(PLUGINS_DIR, 'NX', 'journal_functions.py'), 'r') as g:
                f.writelines(g.readlines())

            ctrlpt_dict = self.get_ctrlpt_dict(zero_col=2)
            for k, ctrlpts in ctrlpt_dict.items():
                new_ctrlpts = np.array(ctrlpts)
                new_ctrlpts *= 36.98030879 * 1000
                ctrlpt_dict[k] = new_ctrlpts.tolist()

            f.write('ctrlpts = {\n')
            for a_name, ctrlpts in ctrlpt_dict.items():
                f.write(f'    "{a_name}": [\n')
                for ctrlpt_set in ctrlpts:
                    f.write(f'        [\n')
                    for ctrlpt in ctrlpt_set:
                        f.write(f'            [{ctrlpt[0]}, {ctrlpt[1]}, {ctrlpt[2]}],\n')
                    f.write(f'        ],\n')
                f.write(f'    ]\n')
            f.write('}\n\n')

            f.write('create_bezier_curve_from_ctrlpts(ctrlpts)\n')

            for k, v in ctrlpt_dict.items():
                for idx, v2 in enumerate(v):
                    v[idx] = v2.tolist()

            save_data(ctrlpt_dict, 'center_profile_2_ctrlpts.json')

    @classmethod
    def generate_from_param_dict(cls, param_dict: dict):
        """
        Reconstruct an MEA from the MEA's JSON-saved param_dict. This is the normal way of saving an airfoil system
        in pymead because it avoids issues associated with serializing portions of the MEA. It also allows for direct
        modification of the save file due to the human-readable JSON save format.

        Parameters
        ==========
        param_dict: dict
          A Python dictionary constructed by using ``json.load`` on a ``.jmea`` file.

        Returns
        =======
        MEA
          An instance of the ``pymead.core.mea.MEA`` class.
        """
        base_params_dict = {k: v['Base'] for k, v in param_dict.items() if isinstance(v, dict) and 'Base' in v.keys()}
        base_params = {}
        for airfoil_name, airfoil_base_dict in base_params_dict.items():
            base_params[airfoil_name] = {}
            for pname, pdict in airfoil_base_dict.items():
                base_params[airfoil_name][pname] = Param.from_param_dict(pdict)
        airfoil_list = []
        for airfoil_name, airfoil_base_dict in base_params.items():
            base = BaseAirfoilParams(airfoil_tag=airfoil_name, **airfoil_base_dict)
            airfoil_list.append(Airfoil(base_airfoil_params=base, tag=airfoil_name))
        mea = cls(airfoils=airfoil_list)  # Constructor overload
        mea.airfoil_graphs_active = param_dict['airfoil_graphs_active']  # set this after MEA instantiation to avoid
        # initializing the graphs
        mea.file_name = param_dict['file_name']
        for a_name, airfoil in mea.airfoils.items():
            ap_order = param_dict[a_name]['anchor_point_order']
            aps = param_dict[a_name]['AnchorPoints']
            for idx, ap_name in enumerate(ap_order):
                if ap_name not in ['te_1', 'le', 'te_2']:
                    ap_dict = aps[ap_name]
                    ap_param_dict = {}
                    for pname, pdict in ap_dict.items():
                        if pname == 'xy':
                            ap_param_dict[pname] = PosParam.from_param_dict(pdict)
                        else:
                            ap_param_dict[pname] = Param.from_param_dict(pdict)

                    # Create an AnchorPoint from the saved parameter dictionary:
                    ap = AnchorPoint(airfoil_tag=a_name, tag=ap_name, previous_anchor_point=ap_order[idx - 1],
                                     **ap_param_dict)

                    # Now, insert the AnchorPoint into the Airfoil
                    airfoil.insert_anchor_point(ap)

            for ap_name, fp_list in param_dict[a_name]['free_point_order'].items():
                fps = param_dict[a_name]['FreePoints'][ap_name]
                for idx, fp_name in enumerate(fp_list):
                    fp_dict = fps[fp_name]
                    fp_param_dict = {}
                    for pname, pdict in fp_dict.items():
                        fp_param_dict[pname] = PosParam.from_param_dict(pdict)

                    previous_fp = fp_list[idx - 1] if idx > 0 else None
                    # Create a FreePoint from the saved parameter dictionary:
                    fp = FreePoint(airfoil_tag=a_name, tag=fp_name, previous_anchor_point=ap_name,
                                   previous_free_point=previous_fp, **fp_param_dict)

                    # Now, insert the FreePoint into the Airfoil
                    airfoil.insert_free_point(fp)

        for custom_name, custom_param in param_dict['Custom'].items():
            custom_param_dict = {}
            temp_dict = {'value': custom_param['_value']}
            for attr_name, attr_value in custom_param.items():
                if attr_name in ['bounds', 'active', 'func_str', 'name']:
                    temp_dict[attr_name] = attr_value
            custom_param_dict[custom_name] = deepcopy(temp_dict)  # Need deepcopy here?
            mea.add_custom_parameters(custom_param_dict)

        mea.assign_names_to_params_in_param_dict()

        mea.user_mods = {}
        for f in INCLUDE_FILES:
            name = os.path.split(f)[-1]  # get the name of the file without the directory
            name_no_ext = os.path.splitext(name)[-2]  # get the name of the file without the .py extension
            spec = importlib.util.spec_from_file_location(name_no_ext, f)
            mea.user_mods[name_no_ext] = importlib.util.module_from_spec(spec)  # generate the module from the name
            spec.loader.exec_module(mea.user_mods[name_no_ext])  # compile and execute the module

        def f(d, key, value):
            if isinstance(value, Param) or isinstance(value, PosParam):
                value.function_dict['name'] = value.name.split('.')[-1]
                value.update()

        dben = benedict.benedict(mea.param_dict)
        dben.traverse(f)

        for a in mea.airfoils.values():
            a.update()

        return mea

    def remove_airfoil_graphs(self):
        """
        Removes the airfoil graph from each airfoil in the system.
        """
        self.airfoil_graphs_active = False
        for a in self.airfoils.values():
            a.airfoil_graph = None
