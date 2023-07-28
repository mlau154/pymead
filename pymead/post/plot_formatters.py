import os
import itertools

import matplotlib.pyplot as plt


def format_axis_scientific(ax: plt.Axes, font_family: str = "serif", font_size: int = 16):
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(font_size)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(font_size)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", left=True, bottom=True, top=True, right=True)
    ax.tick_params(which="major", direction="in", left=True, bottom=True, top=True, right=True)


def show_save_fig(fig: plt.Figure, save_base_dir: str, file_name_stub: str, show: bool = False, save: bool = True,
                  show_first: bool = True, save_ext: tuple = (".png", ".svg", ".pdf")):

    def show_plot():
        plt.show()

    def save_plot():
        for ext in save_ext:
            save_kwargs = {}
            if ext in [".png", ".jpg"]:
                save_kwargs["dpi"] = 600
            fig.savefig(os.path.join(save_base_dir, file_name_stub + ext), bbox_inches="tight", **save_kwargs)

    if show_first:
        if show:
            show_plot()
        if save:
            save_plot()
    else:
        if save:
            save_plot()
        if show:
            show_plot()


def legend_entry_flip(items, ncol):
    """Solution for legend entry flipping from StackOverflow user Avaris @ https://stackoverflow.com/a/10101532"""
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
