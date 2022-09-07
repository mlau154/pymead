from pyairpar.core.airfoil import Airfoil
import typing


class MEA:
    """
    ### Description:

    Class for multi-element airfoils.
    """
    def __init__(self, airfoils: Airfoil or typing.List[Airfoil, ...] or None = None):
        self.airfoils = []
        if not isinstance(airfoils, list):
            if airfoils is not None:
                self.add_airfoil(airfoils, 0)
        else:
            for idx, airfoil in enumerate(airfoils):
                self.add_airfoil(airfoil, idx)

    def add_airfoil(self, airfoil: Airfoil, idx: int):
        if airfoil.tag is None:
            airfoil.tag = f'Airfoil {idx}'
        self.airfoils.append(airfoil)

    def get_index_from_tag(self, tag):
        return next((idx for idx, airfoil in enumerate(self.airfoils) if airfoil.tag == tag))

    def add_inter_airfoil_constraint(self, tag1: str, tag2: str):
        idx1 = self.get_index_from_tag(tag1)
        idx2 = self.get_index_from_tag(tag2)


if __name__ == '__main__':
    from pyairpar.core.base_airfoil_params import BaseAirfoilParams
    from pyairpar.core.param import Param
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
