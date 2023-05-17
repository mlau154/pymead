import matplotlib.pyplot as plt
from pymead.core.airfoil import Airfoil
from pymead.core.anchor_point import AnchorPoint
from pymead.core.param import Param
from math import pi


def main():
    a = Airfoil()
    ap = AnchorPoint(x=Param(0.4), y=Param(-0.08), tag="AP1", L=Param(0.02), R=Param(20.0), r=Param(0.5),
                     phi=Param(0.0), psi1=Param(pi / 2), psi2=Param(pi / 2), previous_anchor_point='le')
    a.insert_anchor_point(ap)
    a.update()
    a.anchor_points[2].L.value = 0.1
    a.update()
    ap = AnchorPoint(x=Param(0.45), y=Param(0.08), tag="AP0", L=Param(0.04), R=Param(10.0), r=Param(0.55),
                     phi=Param(0.05), psi1=Param(pi / 1.5), psi2=Param(pi / 1.5), previous_anchor_point='te_1')
    a.insert_anchor_point(ap)
    a.update()
    print(f"N = {a.N}")
    print(f"order = {a.anchor_point_order}")
    fig, ax = plt.subplots()
    a.plot_airfoil(ax)
    a.plot_control_points(ax, color="grey", ls='--', marker='x')
    plt.show()


if __name__ == '__main__':
    main()
