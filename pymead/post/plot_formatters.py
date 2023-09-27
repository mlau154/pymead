import os
import itertools

import matplotlib.legend
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def format_axis_scientific(ax: plt.Axes, font_family: str = "serif", font_size: int = 16):
    """
    Formats a Matplotlib ``Axes`` similar to the IEEE style
    """
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


def dark_mode(fig: plt.Figure):
    """
    Sets a dark theme for many elements of a Matplotlib ``Figure``

    Parameters
    ==========
    fig: plt.Figure
        Figure to modify
    """

    def _format_axis(axis):
        axis.set_facecolor(black)
        for spine in axis.spines.values():
            spine.set_color("w")
        axis.xaxis.label.set_color("w")
        axis.yaxis.label.set_color("w")
        axis.tick_params(axis="x", colors="w")
        axis.tick_params(axis="y", colors="w")
        axis.set_title(axis.get_title(), color="w")

    def _format_legend(lg):
        for text in lg.get_texts():
            text.set_color("w")
        frame = lg.get_frame()
        frame.set_facecolor("k")

    ax_list = fig.get_axes()
    black = "#121212"
    for ax in ax_list:

        _format_axis(ax)

        for child in ax.get_children():
            print(f"{child = }")
            if isinstance(child, plt.Annotation):
                if child.get_color() == "black":
                    child.set_color("w")
            if isinstance(child, plt.Axes):
                _format_axis(child)
                for grandchild in child.get_children():
                    print(f"{grandchild = }")
                    if isinstance(grandchild, plt.Text):
                        print(f"{grandchild.get_color() = }")
                        if grandchild.get_color() == "black":
                            grandchild.set_color("w")
            if isinstance(child, matplotlib.legend.Legend):
                _format_legend(child)
            if isinstance(child, mpatches.ConnectionPatch):
                child.set_color("w")
            if isinstance(child, mpatches.Rectangle):
                child.set_edgecolor("w")

    fig.set_facecolor(black)
    for leg in fig.legends:
        _format_legend(leg)


def legend_entry_flip(items, ncol):
    """Solution for legend entry flipping from StackOverflow user Avaris @ https://stackoverflow.com/a/10101532"""
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
