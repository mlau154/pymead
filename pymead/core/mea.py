from pymead.core.airfoil import Airfoil
from pymead.core.param import Param
from pymead.utils.dict_recursion import set_all_dict_values
import typing
import benedict


class MEA:
    """
    ### Description:

    Class for multi-element airfoils. Serves as a container for `pymead.core.airfoil.Airfoil`s and adds a few methods
    important for the Graphical User Interface.
    """
    def __init__(self, airfoils: Airfoil or typing.List[Airfoil, ...] or None = None,
                 airfoil_graphs_active: bool = False):
        self.airfoils = {}
        self.param_dict = {}
        self.airfoil_graphs_active = airfoil_graphs_active
        if self.airfoil_graphs_active:
            self.w = None
            self.v = None
        if not isinstance(airfoils, list):
            if airfoils is not None:
                self.add_airfoil(airfoils, 0)
        else:
            for idx, airfoil in enumerate(airfoils):
                self.add_airfoil(airfoil, idx)

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

    def add_airfoil(self, airfoil: Airfoil, idx: int):
        """
        Add an airfoil at index `idx` to the multi-element airfoil container.
        """
        if airfoil.tag is None:
            airfoil.tag = f'A{idx}'
        self.airfoils[airfoil.tag] = airfoil
        self.param_dict[airfoil.tag] = airfoil.param_dicts

        set_all_dict_values(self.param_dict[airfoil.tag])

        self.add_airfoil_graph_to_airfoil(airfoil, idx)

    def add_airfoil_graph_to_airfoil(self, airfoil: Airfoil, idx: int, w=None, v=None):
        """
        Add a `pyqtgraph`-based `pymead.gui.airfoil_graph.AirfoilGraph` to the airfoil at index `int`.
        """
        if self.airfoil_graphs_active:
            from pymead.gui.airfoil_graph import AirfoilGraph
            if w is None:
                if idx == 0:
                    airfoil_graph = AirfoilGraph(airfoil)
                    self.w = airfoil_graph.w
                    self.v = airfoil_graph.v
                else:  # Assign the first airfoil's Graphics Window and ViewBox to each subsequent airfoil
                    airfoil_graph = AirfoilGraph(airfoil,
                                                 w=self.airfoils['A0'].airfoil_graph.w,
                                                 v=self.airfoils['A0'].airfoil_graph.v)
            else:
                airfoil_graph = AirfoilGraph(airfoil, w=w, v=v)
                self.w = w
                self.v = v

            airfoil.airfoil_graph = airfoil_graph

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
