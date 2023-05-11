"""
Images for documentation
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from pymead.core.bezier import Bezier


logging.getLogger().setLevel(logging.INFO)


def save_figure(fig: plt.Figure, ax: plt.Axes, name: str,
                light: bool = True, dark: bool = True, png: bool = True, pdf: bool = True):
    png_kwargs = dict(bbox_inches='tight', dpi=600)
    pdf_kwargs = dict(bbox_inches='tight')
    if light:
        if png:
            fig.savefig(os.path.join('images', f'{name}_light.png'), **png_kwargs)
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_light.pdf'), **pdf_kwargs)
    if dark:
        black = '#121212'
        fig.patch.set_facecolor(black)
        ax.set_facecolor(black)
        for sp in ax.spines.values():
            sp.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if png:
            fig.savefig(os.path.join('images', f'{name}_dark.png'), **png_kwargs)
        if pdf:
            fig.savefig(os.path.join('images', f'{name}_dark.pdf'), **pdf_kwargs)


def cubic_bezier(show: bool, save: bool):
    # Plot settings
    plt.rcParams["font.family"] = "serif"
    color_control_points = "#b3b3b3ff"
    color_curve = "#2cdbcaff"

    # Generate the plot
    fig, ax = plt.subplots(figsize=(8, 3))
    P = np.array([
        [0, 0],
        [0.25, 0.5],
        [0.75, -0.5],
        [1, 0]
    ])
    bez = Bezier(P=P, nt=150)
    bez.plot_curve(ax, color=color_curve, lw=2, label='curve')
    bez.plot_control_point_skeleton(ax, color=color_control_points, ls='--', marker='*', lw=1.5,
                                    label='control points', markersize=10)
    ax.legend()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if show:
        plt.show()
    if save:
        save_figure(fig, ax, "cubic_bezier")


def main():
    show = True
    save = True
    cubic_bezier(show, save)


if __name__ == '__main__':
    main()
