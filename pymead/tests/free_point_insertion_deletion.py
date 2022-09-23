import matplotlib.pyplot as plt
from pymead.core.airfoil import Airfoil
from pymead.core.free_point import FreePoint
from pymead.core.param import Param


def main():
    a = Airfoil()
    fp = FreePoint(x=Param(0.4), y=Param(-0.08), previous_anchor_point='le', previous_free_point=None)
    a.insert_free_point(fp)
    a.update()
    fp2 = FreePoint(x=Param(0.7), y=Param(-0.05), previous_anchor_point='le', previous_free_point="FP0")
    a.insert_free_point(fp2)
    a.update()
    fp3 = FreePoint(x=Param(0.5), y=Param(-0.1), previous_anchor_point='le', previous_free_point="FP0")
    a.insert_free_point(fp3)
    a.update()
    a.delete_free_point('FP1', 'le')
    a.update()
    fp4 = FreePoint(x=Param(0.55), y=Param(0.1), previous_anchor_point='te_1', previous_free_point=None)
    a.insert_free_point(fp4)
    a.update()
    # print(f"control_point_array = {a.control_point_array}")
    fig, ax = plt.subplots()
    a.plot_airfoil(ax)
    a.plot_control_points(ax, color="grey", ls='--', marker='x')
    plt.show()


if __name__ == '__main__':
    main()
