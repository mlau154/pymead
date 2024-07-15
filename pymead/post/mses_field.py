import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np
import os
from pymead.analysis.read_aero_data import read_grid_stats_from_mses, read_field_from_mses, \
    read_streamline_grid_from_mses, read_Mach_from_mses_file
from pymead.analysis.read_aero_data import flow_var_idx
from pymead.core.transformation import Transformation2D

flow_var_label = {'M': 'Mach Number',
                  'Cp': 'Pressure Coefficient',
                  'p': 'Static Pressure (p / p<sub>\u221e</sub>)',
                  'rho': 'Density (\u03c1/\u03c1<sub>\u221e</sub>)',
                  'u': 'Velocity-x (u/V<sub>\u221e</sub>)',
                  'v': 'Velocity-y (v/V<sub>\u221e</sub>)',
                  'V': 'Velocity-mag (V/V<sub>\u221e</sub>)',
                  'q': 'Speed of Sound (q/V<sub>\u221e</sub>)',
                  "Cpt": "Total Pressure / Pinf",
                  "dCpt": "Delta Total Pressure"}


def generate_field_matplotlib(axs: plt.Axes or None,
                              analysis_subdir: str,
                              var: str,
                              cmap_field: mpl_colors.Colormap or str,
                              vmin: float = None,
                              vmax: float = None,
                              arrow_start_J_idx: int = 20,
                              arrow_spacing: int = 60,
                              field_scaling_factor: float = None):
    pcolormesh_handles = {'field': None, 'airfoil': []}
    field_file = os.path.join(analysis_subdir, f'field.{os.path.split(analysis_subdir)[-1]}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{os.path.split(analysis_subdir)[-1]}')
    mses_file = os.path.join(analysis_subdir, f"mses.{os.path.split(analysis_subdir)[-1]}")
    if not os.path.exists(field_file):
        raise OSError(f"Field file {field_file} not found")
    if not os.path.exists(grid_stats_file):
        raise OSError(f"Grid statistics log {grid_stats_file} not found")
    if not os.path.exists(grid_file):
        raise OSError(f"Grid file {grid_file} not found")

    M_inf = read_Mach_from_mses_file(mses_file)
    gam = 1.4
    field = read_field_from_mses(field_file, M_inf=M_inf, gam=gam)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
    if field_scaling_factor:
        x_grid = [x_grid_section * field_scaling_factor for x_grid_section in x_grid]
        y_grid = [y_grid_section * field_scaling_factor for y_grid_section in y_grid]

    flow_var = field[flow_var_idx[var]]

    if vmin is None:
        vmin = np.min(flow_var)
    if vmax is None:
        vmax = np.max(flow_var)

    # Stagnation streamline arrows
    max_arrow_idx = x_grid[0].shape[0] - 2
    if arrow_start_J_idx > max_arrow_idx:
        raise ValueError(f"Starting arrow index too large (greater than max J index of {max_arrow_idx})")
    arrow_idxs = [arrow_start_J_idx]
    while True:
        if arrow_idxs[-1] + arrow_spacing > max_arrow_idx:
            break
        else:
            arrow_idxs.append(arrow_idxs[-1] + arrow_spacing)

    start_idx, end_idx = 0, x_grid[0].shape[1] - 1
    for flow_section_idx in range(grid_stats["numel"] + 1):

        flow_var_section = flow_var[:, start_idx:end_idx]

        args = (x_grid[flow_section_idx], y_grid[flow_section_idx], flow_var_section)
        kwargs = dict(cmap=cmap_field, vmin=vmin, vmax=vmax)

        if axs is None:
            pcolormesh_handles[f'field_{flow_section_idx}'] = plt.pcolormesh(*args, **kwargs)
        else:
            pcolormesh_handles[f'field_{flow_section_idx}'] = axs.pcolormesh(*args, **kwargs)

        if flow_section_idx == 0:
            x_stags = [x_grid[flow_section_idx][:, -1]]
            y_stags = [y_grid[flow_section_idx][:, -1]]
        elif flow_section_idx == grid_stats["numel"]:
            x_stags = [x_grid[flow_section_idx][:, 0]]
            y_stags = [y_grid[flow_section_idx][:, 0]]
        else:
            x_stags = [x_grid[flow_section_idx][:, 0], x_grid[flow_section_idx][:, -1]]
            y_stags = [y_grid[flow_section_idx][:, 0], y_grid[flow_section_idx][:, -1]]

        axs = plt.gca() if axs is None else axs
        for x_stag, y_stag in zip(x_stags, y_stags):
            axs.plot(x_stag, y_stag, color="blue")
            for arrow_idx in arrow_idxs:
                axs.arrow(x_stag[arrow_idx], y_stag[arrow_idx],
                          x_stag[arrow_idx + 1] - x_stag[arrow_idx], y_stag[arrow_idx + 1] - y_stag[arrow_idx],
                          color="blue", width=0.0001, head_width=0.01, zorder=1000)

        if flow_section_idx < grid_stats["numel"]:
            start_idx = end_idx
            end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

    if axs is None:
        return pcolormesh_handles
    else:
        return axs, pcolormesh_handles
