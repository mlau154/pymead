from pymead.core.airfoil import Airfoil
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.utils.dict_recursion import set_all_dict_values, assign_airfoil_tags_to_param_dict, \
    assign_names_to_params_in_param_dict
import typing
import benedict
import numpy as np
import os
from pymead import DATA_DIR
from copy import deepcopy


class MEA:
    """
    ### Description:

    Class for multi-element airfoils. Serves as a container for `pymead.core.airfoil.Airfoil`s and adds a few methods
    important for the Graphical User Interface.
    """
    def __init__(self, param_tree=None, airfoils: Airfoil or typing.List[Airfoil, ...] or None = None,
                 airfoil_graphs_active: bool = False):
        self.airfoils = {}
        self.file_name = None
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

    def __getstate__(self):
        """
        Reimplemented to ensure MEA picklability
        """
        state = self.__dict__.copy()
        if state['v'] is not None:
            state['v'].clear()
        state['w'] = None  # Set unpicklable GraphicsLayoutWidget object to None
        state['v'] = None  # Set unpicklable ViewBox object to None
        return state

    def add_airfoil(self, airfoil: Airfoil, idx: int, param_tree, w=None, v=None):
        """
        Add an airfoil at index `idx` to the multi-element airfoil container.
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
            self.add_airfoil_graph_to_airfoil(airfoil, idx, param_tree, w=w, v=v)

        dben = benedict.benedict(self.param_dict)
        for k in dben.keypaths():
            param = dben[k]
            if isinstance(param, Param):
                if param.mea is None:
                    param.mea = self
                if param.mea.param_tree is None:
                    param.mea.param_tree = self.param_tree

    def remove_airfoil(self):
        # TODO: implement airfoil removal feature
        pass

    def assign_names_to_params_in_param_dict(self):
        assign_names_to_params_in_param_dict(self.param_dict)

    def add_airfoil_graph_to_airfoil(self, airfoil: Airfoil, idx: int, param_tree, w=None, v=None):
        """
        Add a `pyqtgraph`-based `pymead.gui.airfoil_graph.AirfoilGraph` to the airfoil at index `int`.
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
                                             v=self.airfoils['A0'].airfoil_graph.v)
                # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
                airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
        else:
            print("Creating new AirfoilGraph!")
            airfoil_graph = AirfoilGraph(airfoil, w=w, v=v)
            # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
            airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
            self.w = w
            self.v = v

        airfoil_graph.param_tree = param_tree
        airfoil.airfoil_graph = airfoil_graph

    def extract_parameters(self, write_to_txt_file: bool = False):
        parameter_list = []
        norm_value_list = []

        def check_for_bounds_recursively(d: dict, bounds_error_=False):
            for k_, v in d.items():
                if not bounds_error_:
                    if isinstance(v, dict):
                        bounds_error_, param_name_ = check_for_bounds_recursively(v, bounds_error_)
                    else:
                        if isinstance(v, Param):
                            if v.active and not v.linked:
                                if v.bounds[0] == -np.inf or v.bounds[0] == np.inf or v.bounds[1] == -np.inf or v.bounds[1] == np.inf:
                                    bounds_error_ = True
                                    return bounds_error_, v.name
                        else:
                            raise ValueError('Found value in dictionary not of type \'Param\'')
                else:
                    return bounds_error_, None
            return bounds_error_, None

        def extract_parameters_recursively(d: dict):
            for k_, v in d.items():
                if isinstance(v, dict):
                    extract_parameters_recursively(v)
                else:
                    if isinstance(v, Param):
                        if v.active and not v.linked:
                            norm_value_list.append((v.value - v.bounds[0]) / (v.bounds[1] - v.bounds[0]))
                            parameter_list.append(v)
                    else:
                        raise ValueError('Found value in dictionary not of type \'Param\'')

        bounds_error, param_name = check_for_bounds_recursively(self.param_dict)
        if bounds_error:
            error_message = f'Bounds must be set for each active and unlinked parameter for parameter extraction (at ' \
                            f'least one infinite bound found for {param_name})'
            print(error_message)
            return error_message
        else:
            extract_parameters_recursively(self.param_dict)
            if write_to_txt_file:
                np.savetxt(os.path.join(DATA_DIR, 'parameter_list.dat'), np.array(norm_value_list))
            # parameter_list = np.loadtxt(os.path.join(DATA_DIR, 'parameter_list.dat'))
            # self.update_parameters(parameter_list)
            # fig_.savefig(os.path.join(DATA_DIR, 'test_airfoil.png'))
            return norm_value_list, parameter_list

    def update_parameters(self, norm_value_list: list or np.ndarray):

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
                        if isinstance(v, Param):
                            if v.active and not v.linked:
                                if v.bounds[0] == -np.inf or v.bounds[0] == np.inf or v.bounds[1] == -np.inf or \
                                        v.bounds[1] == np.inf:
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
                    if isinstance(v, Param):
                        if v.active and not v.linked:
                            v.value = norm_value_list[list_counter] * (v.bounds[1] - v.bounds[0]) + v.bounds[0]
                            # v.update()
                            list_counter += 1
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
            update_parameters_recursively(self.param_dict, 0)

            for a_tag, airfoil in self.airfoils.items():
                airfoil.update()
                if self.airfoil_graphs_active:
                    airfoil.airfoil_graph.data['pos'] = airfoil.control_point_array
                    airfoil.airfoil_graph.updateGraph()
                    airfoil.airfoil_graph.plot_change_recursive(
                        airfoil.airfoil_graph.airfoil_parameters.child(a_tag).children())

    def deactivate_airfoil_matching_params(self, target_airfoil: str):
        def deactivate_recursively(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    # If the dictionary key is equal to the target_airfoil_str, stop the recursion for this branch,
                    # otherwise continue with the recursion:
                    if k != target_airfoil:
                        deactivate_recursively(v)
                elif isinstance(v, Param):
                    if v.active:
                        v.active = False
                        v.deactivated_for_airfoil_matching = True
                else:
                    raise ValueError('Found value in dictionary not of type \'Param\'')

        deactivate_recursively(self.param_dict)
        deactivate_target_params = ['dx', 'dy', 'alf', 'c', 't_te', 'r_te', 'phi_te']
        for p_str in deactivate_target_params:
            p = self.airfoils[target_airfoil].param_dicts['Base'][p_str]
            if p.active:
                p.active = False
                p.deactivated_for_airfoil_matching = True

    def activate_airfoil_matching_params(self, target_airfoil: str):
        def activate_recursively(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    # If the dictionary key is equal to the target_airfoil_str, stop the recursion for this branch,
                    # otherwise continue with the recursion:
                    if k != target_airfoil:
                        activate_recursively(v)
                elif isinstance(v, Param):
                    if v.deactivated_for_airfoil_matching:
                        v.active = True
                        v.deactivated_for_airfoil_matching = False
                else:
                    raise ValueError('Found value in dictionary not of type \'Param\'')

        activate_recursively(self.param_dict)
        activate_target_params = ['dx', 'dy', 'alf', 'c']
        for p_str in activate_target_params:
            p = self.airfoils[target_airfoil].param_dicts['Base'][p_str]
            if p.deactivated_for_airfoil_matching:
                p.active = True
                p.deactivated_for_airfoil_matching = False

    def calculate_max_x_extent(self):
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
        Used in `pymead.core.parameter_tree.MEAParamTree` to update the equation variable AutoCompleter
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

    @classmethod
    def generate_from_param_dict(cls, param_dict: dict):
        """Reconstruct an MEA from the MEA's JSON-saved param_dict"""
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
        mea = cls(airfoils=airfoil_list)
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
                        fp_param_dict[pname] = Param.from_param_dict(pdict)

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

        return mea


if __name__ == '__main__':
    from pymead.core.base_airfoil_params import BaseAirfoilParams
    from matplotlib.pyplot import subplots, show
    airfoil1 = Airfoil()
    airfoil2 = Airfoil(base_airfoil_params=BaseAirfoilParams(dy=Param(0.2)))
    mea_ = MEA(airfoils=[airfoil1, airfoil2])
    fig, axs = subplots()
    colors = ['cornflowerblue', 'indianred']
    for _idx, _airfoil in enumerate(mea_.airfoils):
        _airfoil.plot_airfoil(axs, color=colors[_idx], label=_airfoil.tag)
    axs.set_aspect('equal')
    axs.legend()
    show()
