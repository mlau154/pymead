import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np
import os
from pymead.analysis.read_aero_data import read_grid_stats_from_mses


flow_var_idx = {'M': 7, 'Cp': 8, 'p': 5, 'rho': 4, 'u': 2, 'v': 3, 'q': 6}

flow_var_label = {'M': 'Mach Number',
                  'Cp': 'Pressure Coefficient',
                  'p': 'Static Pressure (p / p<sub>\u221e</sub>)',
                  'rho': 'Density (\u03c1/\u03c1<sub>\u221e</sub>)',
                  'u': 'Velocity-x (u/V<sub>\u221e</sub>)',
                  'v': 'Velocity-y (v/V<sub>\u221e</sub>)',
                  'q': 'Speed of Sound (q/V<sub>\u221e</sub>)'}


def generate_field_matplotlib(axs: plt.Axes or None, analysis_subdir: str, var: str, cmap_field: mpl_colors.Colormap or str ,
                              cmap_airfoil: mpl_colors.Colormap or str, shading: str = 'gouraud', vmin: float = None,
                              vmax: float = None):
    pcolormesh_handles = {'field': None, 'airfoil': []}
    field_file = os.path.join(analysis_subdir, f'field.{os.path.split(analysis_subdir)[-1]}')
    grid_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    if not os.path.exists(field_file):
        raise OSError(f"Field file {field_file} not found")
    if not os.path.exists(grid_file):
        raise OSError(f"Grid statistics log {grid_file} not found")

    data = np.loadtxt(field_file, skiprows=2)
    grid = read_grid_stats_from_mses(grid_file)

    with open(field_file, 'r') as f:
        lines = f.readlines()

    n_streamlines = 0
    for line in lines:
        if line == '\n':
            n_streamlines += 1

    n_streamwise_lines = int(data.shape[0] / n_streamlines)

    x = data[:, 0].reshape(n_streamlines, n_streamwise_lines).T
    y = data[:, 1].reshape(n_streamlines, n_streamwise_lines).T

    flow_var = data[:, flow_var_idx[var]].reshape(n_streamlines, n_streamwise_lines).T

    if vmin is None:
        vmin = np.min(flow_var)
    if vmax is None:
        vmax = np.max(flow_var)

    if shading == 'auto':
        interpolated_flow_var = np.zeros(shape=(x.shape[0] - 1, x.shape[1] - 1))
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                interpolated_flow_var[i, j] = np.mean(flow_var[i:i + 2, j:j + 2])
        plot_flow_var = interpolated_flow_var

        if axs is None:
            pcolormesh_handles['field'] = plt.pcolormesh(x, y, plot_flow_var, cmap=cmap_field, shading=shading,
                                                         vmin=vmin, vmax=vmax)
        else:
            pcolormesh_handles['field'] = axs.pcolormesh(x, y, plot_flow_var, cmap=cmap_field, shading=shading,
                                                         vmin=vmin, vmax=vmax)
    else:
        plot_flow_var = flow_var

        if axs is None:
            pcolormesh_handles['field'] = plt.pcolormesh(x, y, plot_flow_var, cmap=cmap_field, shading=shading,
                                                         vmin=vmin, vmax=vmax)
        else:
            pcolormesh_handles['field'] = axs.pcolormesh(x, y, plot_flow_var, cmap=cmap_field, shading=shading,
                                                         vmin=vmin, vmax=vmax)

    for el in range(grid['numel']):
        offset = grid['numel'] - el
        x_gray = x[:, grid['Jside2'][el] - 1 - offset:grid['Jside1'][el] - offset]
        y_gray = y[:, grid['Jside2'][el] - 1 - offset:grid['Jside1'][el] - offset]
        v_gray = np.zeros(shape=x_gray.shape)[:-1, :-1]
        if axs is None:
            pcolormesh_handles['airfoil'].append(plt.pcolormesh(x_gray, y_gray, v_gray, cmap=cmap_airfoil))
        else:
            pcolormesh_handles['airfoil'].append(axs.pcolormesh(x_gray, y_gray, v_gray, cmap=cmap_airfoil))

    if axs is None:
        return pcolormesh_handles
    else:
        return axs, pcolormesh_handles
