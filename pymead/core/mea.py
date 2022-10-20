from pymead.core.airfoil import Airfoil
from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.core.param import Param
from pymead.utils.dict_recursion import set_all_dict_values, assign_airfoil_tags_to_param_dict
import typing
import benedict
import numpy as np


class MEA:
    """
    ### Description:

    Class for multi-element airfoils. Serves as a container for `pymead.core.airfoil.Airfoil`s and adds a few methods
    important for the Graphical User Interface.
    """
    def __init__(self, param_tree, airfoils: Airfoil or typing.List[Airfoil, ...] or None = None,
                 airfoil_graphs_active: bool = False):
        self.airfoils = {}
        self.param_dict = {}
        self.airfoil_graphs_active = airfoil_graphs_active
        self.te_thickness_edit_mode = False
        if self.airfoil_graphs_active:
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
        print(f"MEA __getstate__ called")
        state = self.__dict__.copy()
        print(f"state v = {state['v']}")
        print(f"state = {state}")
        if state['v'] is not None:
            state['v'].clear()
        state['w'] = None  # Set unpicklable GraphicsLayoutWidget object to None
        state['v'] = None  # Set unpicklable ViewBox object to None
        return state

    # def __setstate__(self, state):
    #     """
    #     Reimplemented to re-add GraphicsLayoutWidget and ViewBox objects to airfoils after unpickling
    #     """
    #     print(f"MEA __setstate__ called")
    #     self.__dict__.update(state)
    #     for idx, airfoil in enumerate(self.airfoils):
    #         self.add_airfoil_graph_to_airfoil(airfoil, idx)

    def add_airfoil(self, airfoil: Airfoil, idx: int, param_tree):
        """
        Add an airfoil at index `idx` to the multi-element airfoil container.
        """
        if airfoil.tag is None:
            airfoil.tag = f'A{idx}'
        # print(f"Adding mea to airfoil {airfoil.tag}")
        airfoil.mea = self
        self.airfoils[airfoil.tag] = airfoil
        self.param_dict[airfoil.tag] = airfoil.param_dicts
        print(f"param_dict = {self.param_dict}")

        set_all_dict_values(self.param_dict[airfoil.tag])

        assign_airfoil_tags_to_param_dict(self.param_dict[airfoil.tag], airfoil_tag=airfoil.tag)

        if self.airfoil_graphs_active:
            self.add_airfoil_graph_to_airfoil(airfoil, idx, param_tree)

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
            airfoil_graph = AirfoilGraph(airfoil, w=w, v=v)
            # print(f"setting te_thickness_edit_mode of airfoil {airfoil.tag} to {self.te_thickness_edit_mode}")
            airfoil_graph.te_thickness_edit_mode = self.te_thickness_edit_mode
            self.w = w
            self.v = v

        airfoil_graph.param_tree = param_tree
        airfoil.airfoil_graph = airfoil_graph

    def extract_parameters(self):
        parameter_list = []

        def extract_parameters_recursively(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    extract_parameters_recursively(v)
                else:
                    if isinstance(v, Param):
                        if v.active and not v.linked:
                            if v.bounds[0] == -np.inf or v.bounds[0] == np.inf or v.bounds[1] == -np.inf or v.bounds[1] == np.inf:
                                raise ValueError('Bounds for parameter must be set to finite values for extraction')
                            else:
                                if self.xy_update_rule(v):
                                    parameter_list.append((v.value - v.bounds[0]) / (v.bounds[1] - v.bounds[0]))
                    else:
                        raise ValueError('Found value in dictionary not of type \'Param\'')

        extract_parameters_recursively(self.param_dict)
        print(f"parameter_list = {parameter_list}")
        return parameter_list

    @staticmethod
    def xy_update_rule(p: Param):
        if p.x or p.y or p.xp or p.yp:
            if p.free_point:
                fp_or_ap = p.free_point
            elif p.anchor_point:
                fp_or_ap = p.anchor_point
            else:
                raise ValueError(f'Neither FreePoint nor AnchorPoint was found for parameter {p}')
            if fp_or_ap.more_than_one_xy_linked_or_inactive():
                return False
            if fp_or_ap.x_or_y_linked_or_inactive() and (p.xp or p.yp):
                return False
            if fp_or_ap.xp_or_yp_linked_or_inactive() and (p.x or p.y):
                return False
        return True

    # @staticmethod
    # def more_than_one_xy_linked_or_inactive(fp_or_ap: FreePoint or AnchorPoint):
    #     linked_or_inactive_counter = 0
    #     for xy in ['x', 'y', 'xp', 'yp']:
    #         if getattr(fp_or_ap, xy).linked or not getattr(fp_or_ap, xy).active:
    #             linked_or_inactive_counter += 1
    #     if linked_or_inactive_counter > 1:
    #         return True
    #     else:
    #         return False
    #
    # @staticmethod
    # def x_or_y_linked_or_inactive(fp_or_ap: FreePoint or AnchorPoint):
    #     for xy in ['x', 'y']:
    #         if getattr(fp_or_ap, xy).linked or not getattr(fp_or_ap, xy).active:
    #             return True
    #     return False
    #
    # @staticmethod
    # def xp_or_yp_linked_or_inactive(fp_or_ap: FreePoint or AnchorPoint):

    def add_custom_parameters(self, params: dict):
        if 'Custom' not in self.param_dict.keys():
            self.param_dict['Custom'] = {}
        for k, v in params.items():
            self.param_dict['Custom'][k] = Param(**v)
            self.param_dict['Custom'][k].param_dict = self.param_dict

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


if __name__ == '__main__':
    from pymead.core.base_airfoil_params import BaseAirfoilParams
    from pymead.core.param import Param
    from matplotlib.pyplot import subplots, show
    airfoil1 = Airfoil()
    airfoil2 = Airfoil(base_airfoil_params=BaseAirfoilParams(dy=Param(0.2)))
    mea = MEA(airfoils=[airfoil1, airfoil2])
    fig, axs = subplots()
    colors = ['cornflowerblue', 'indianred']
    for _idx, _airfoil in enumerate(mea.airfoils):
        _airfoil.plot_airfoil(axs, color=colors[_idx], label=_airfoil.tag)
    axs.set_aspect('equal')
    axs.legend()
    show()
