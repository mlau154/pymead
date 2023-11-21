from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


class ConstraintSolver:
    def __init__(self):
        self.solve_equation()

    def add_distance_constraint(self, indices):
        pass

    def solve_equation(self):

        def equations(x: np.ndarray, x0, y0, dist, ang):
            return np.array([
                (x0 - x[0]) ** 2 + (y0 - x[1]) ** 2 - dist**2,
                np.arctan2(x[1] - y0, x[0] - x0) - ang
            ])

        _x0 = 3
        _y0 = 2
        _dist = 5
        _ang = 0.5 + np.pi
        out = fsolve(equations, np.array([1.0, 1.0]), args=(_x0, _y0, _dist, _ang))
        print(f"{out = }")
        plt.plot(np.array([_x0, out[0]]), np.array([_y0, out[1]]), ls="--", marker="o", color="cornflowerblue",
                 mec="gray", mfc="indianred")
        plt.show()


if __name__ == "__main__":
    cs = ConstraintSolver()
