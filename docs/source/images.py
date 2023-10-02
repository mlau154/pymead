"""
Images for documentation
"""
import typing
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

from pymead.core.bezier import Bezier
from pymead.tutorials import curvature_comb_plotting
from pymead.post.plot_formatters import dark_mode


def save_figure(fig: plt.Figure, axs: plt.Axes, name: str,
                light: bool = True, dark: bool = True, png: bool = True, pdf: bool = True):
    png_kwargs = dict(bbox_inches='tight', dpi=600)
    pdf_kwargs = dict(bbox_inches='tight')
    if light:
        if png:
            fig.savefig(os.path.join('images', f'{name}_light.png'), **png_kwargs)
            print(f"Saved {os.path.join('images', f'{name}_light.png')}")
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_light.pdf'), **pdf_kwargs)
            print(f"Saved {os.path.join('images', f'{name}_light.pdf')}")
    if dark:
        black = '#121212'
        fig.patch.set_facecolor(black)
        if len(fig.legends) > 0:
            pass
        axs = [axs] if not isinstance(axs, typing.Iterable) else axs
        for ax in axs:
            ax.set_facecolor(black)
            for sp in ax.spines.values():
                sp.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            if ax.title:
                ax.title.set_color("white")
        if png:
            fig.savefig(os.path.join('images', f'{name}_dark.png'), **png_kwargs)
            print(f"Saved {os.path.join('images', f'{name}_dark.png')}")
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_dark.pdf'), **pdf_kwargs)
            print(f"Saved {os.path.join('images', f'{name}_dark.pdf')}")


def cubic_bezier(show: bool, save: bool):
    # Plot settings
    plt.rcParams["font.family"] = "serif"
    color_control_points = "#b3b3b3ff"
    color_curve = "#2cdbcaff"

    # Generate the plot
    fig, axs = plt.subplots(figsize=(8, 3))
    P = np.array([
        [0, 0],
        [0.25, 0.5],
        [0.75, -0.5],
        [1, 0]
    ])
    bez = Bezier(P=P, nt=150)
    bez.plot_curve(axs, color=color_curve, lw=2, label='curve')
    bez.plot_control_point_skeleton(axs, color=color_control_points, ls='--', marker='*', lw=1.5,
                                    label='control points', markersize=10)
    axs.legend()
    axs.set_xlabel(r"$x$")
    axs.set_ylabel(r"$y$")

    if show:
        plt.show()
    if save:
        save_figure(fig, axs, "cubic_bezier")


def cubic_bezier_animated():
    # Plot settings
    plt.rcParams["font.family"] = "serif"
    color_control_points = "#b3b3b3ff"
    color_curve = "#2cdbcaff"

    # Generate the plot
    fig, axs = plt.subplots(figsize=(8, 3))

    # Generate the control points for the Bezier curve
    P = np.array([
        [0, 0],
        [0.25, 0.5],
        [0.75, -0.5],
        [1, 0]
    ])

    # Generate the path for the second control point to follow in the animation
    t = np.linspace(np.pi / 2, -3/2 * np.pi, 100)
    r = 0.15  # radius of the circle
    xc = 0.25  # x-location of the center of the circle
    yc = 0.5 - r  # y-location of the center of the circle (positioned directly below the starting point of the cp)
    x = r * np.cos(t) + xc
    y = r * np.sin(t) + yc

    bez = Bezier(P=P, nt=150)
    curve, = axs.plot(bez.x, bez.y, color=color_curve, lw=2, label="curve")
    cps, = axs.plot(P[:, 0], P[:, 1], color=color_control_points, ls="--", marker="*", lw=1.5, label="control points",
                    markersize=10)
    fig.set_tight_layout("tight")

    def init_func():
        curve.set_data([], [])
        cps.set_data([], [])
        axs.legend()
        axs.set_xlabel(r"$x$")
        axs.set_ylabel(r"$y$")
        return [curve, cps]

    def func(frame):
        P[1, 0] = x[frame + 1]
        P[1, 1] = y[frame + 1]
        b = Bezier(P=P, nt=150)
        curve.set_data(b.x, b.y)
        cps.set_data(P[:, 0], P[:, 1])
        return [curve, cps]

    ani = FuncAnimation(fig=fig, frames=len(t) - 1, func=func, init_func=init_func, interval=20, blit=True)
    ani.save("gui_help/images/cubic_bezier_animated_light.gif")

    dark_mode(fig)
    ani = FuncAnimation(fig=fig, frames=len(t) - 1, func=func, init_func=init_func, interval=20, blit=True)
    ani.save("gui_help/images/cubic_bezier_animated_dark.gif")


def main():
    show = False
    save = True
    # cubic_bezier(show, save)
    # fig, axs = curvature_comb_plotting.main()
    # save_figure(fig, axs, "curvature_comb", dark=False)
    # fig, axs = curvature_comb_plotting.main(dark=True)
    # save_figure(fig, axs, "curvature_comb", light=False)
    cubic_bezier_animated()


if __name__ == '__main__':
    main()
