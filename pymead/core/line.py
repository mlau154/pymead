from pymead.core.parametric_curve import ParametricCurve
import numpy as np
from math import tan


class FiniteLine(ParametricCurve):

    def __init__(self, x1, y1, x2, y2, nt: int = 2, t=None):
        self.nt = nt

        if t is None:
            t = np.linspace(0.0, 1.0, self.nt)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.m = (y2 - y1) / (x2 - x1)  # calculate the slope of the line
        self.theta = np.arctan2(self.m, 1)  # get the angle of the line in radians

        x = x1 + t * (x2 - x1) * np.cos(self.theta)
        y = y1 + t * (y2 - y1) * np.sin(self.theta)
        px = np.ones(shape=t.shape)
        py = self.m * np.ones(shape=t.shape)
        ppx = np.zeros(shape=t.shape)
        ppy = np.zeros(shape=t.shape)
        k = np.zeros(shape=t.shape)
        R = np.inf * np.zeros(shape=t.shape)

        super().__init__(t, x, y, px, py, ppx, ppy, k, R)

    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        comb_heads_x = self.x
        comb_heads_y = self.y
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        self.comb_tails = np.column_stack((self.x, self.y))[:-1:interval, :]
        self.comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        self.comb_tails = np.row_stack((self.comb_tails, np.array([self.x[-1], self.y[-1]])))
        self.comb_heads = np.row_stack((self.comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1]])))


class InfiniteLine:
    """This class is not designed for plotting, but merely as a container for line parameters. It also provides some
    convenience functions, especially useful in the 'mirror' function in the GUI."""
    def __init__(self, x1=None, y1=None, x2=None, y2=None, m=None, theta_rad=None, theta_deg=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.m = m
        self.theta_deg = theta_deg
        self.theta_rad = theta_rad
        self.update()

    def update(self):
        # Slope input handling
        if self.m is None:
            if not self.theta_rad and not self.theta_deg:
                self.m = (self.y2 - self.y1) / (self.x2 - self.x1)
            elif self.theta_rad and not self.theta_deg:
                self.m = np.tan(self.theta_rad)
            elif not self.theta_rad and self.theta_deg:
                self.m = np.tan(np.deg2rad(self.theta_deg))

        # Angle input handling
        if not self.theta_rad and not self.theta_deg:
            self.theta_rad = np.arctan2(self.m, 1)
            self.theta_deg = np.rad2deg(self.theta_rad)
        elif self.theta_rad and not self.theta_deg:
            self.theta_deg = np.rad2deg(self.theta_rad)
        elif not self.theta_rad and self.theta_deg:
            self.theta_rad = np.deg2rad(self.theta_deg)
        else:
            pass

        if not self.x2 or not self.y2:
            self.x2 = self.x1 + np.cos(self.theta_rad)
            self.y2 = self.y1 + np.sin(self.theta_rad)

    def get_standard_form_coeffs(self):
        return {'A': -self.m, 'B': 1, 'C': self.m * self.x2 - self.y2}


def _test():
    inf_line = InfiniteLine(x1=0.5, y1=0.2, theta_deg=30)
    print(f"Standard form coefficients = {inf_line.get_standard_form_coeffs()}")


if __name__ == '__main__':
    _test()
