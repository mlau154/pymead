"""
Images for documentation
"""
import typing

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from pymead.core.bezier import Bezier
from pymead.tutorials import curvature_comb_plotting


logging.getLogger().setLevel(logging.INFO)


def save_figure(fig: plt.Figure, axs: plt.Axes, name: str,
                light: bool = True, dark: bool = True, png: bool = True, pdf: bool = True):
    png_kwargs = dict(bbox_inches='tight', dpi=600)
    pdf_kwargs = dict(bbox_inches='tight')
    if light:
        if png:
            fig.savefig(os.path.join('images', f'{name}_light.png'), **png_kwargs)
            logging.info(f"Saved {os.path.join('images', f'{name}_light.png')}")
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_light.pdf'), **pdf_kwargs)
            logging.info(f"Saved {os.path.join('images', f'{name}_light.pdf')}")
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
            logging.info(f"Saved {os.path.join('images', f'{name}_dark.png')}")
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_dark.pdf'), **pdf_kwargs)
            logging.info(f"Saved {os.path.join('images', f'{name}_dark.pdf')}")


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


def main():
    show = False
    save = True
    cubic_bezier(show, save)
    fig, axs = curvature_comb_plotting.main()
    save_figure(fig, axs, "curvature_comb", dark=False)
    fig, axs = curvature_comb_plotting.main(dark=True)
    save_figure(fig, axs, "curvature_comb", light=False)


if __name__ == '__main__':
    main()
