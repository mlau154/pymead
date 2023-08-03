import subprocess
import os
import typing
from copy import deepcopy
import time

import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator, NearestNDInterpolator, LinearNDInterpolator
from shapely.geometry import LineString, Point, MultiPoint

from pymead.analysis.read_aero_data import read_aero_data_from_xfoil, read_Cp_from_file_xfoil, read_bl_data_from_mses, \
    read_forces_from_mses, read_grid_stats_from_mses, read_field_from_mses, read_streamline_grid_from_mses, \
    flow_var_idx, convert_blade_file_to_3d_array
from pymead.utils.file_conversion import convert_ps_to_svg
from pymead.utils.geometry import check_airfoil_self_intersection, convert_numpy_array_to_shapely_LineString
from pymead.utils.read_write_files import write_tuple_tuple_to_file

SVG_PLOTS = ['Mach_contours', 'grid', 'grid_zoom']
SVG_SETTINGS_TR = {
    SVG_PLOTS[0]: 'Mach',
    SVG_PLOTS[1]: 'Grid',
    SVG_PLOTS[2]: 'Grid_Zoom',
}


def update_mses_settings_from_stencil(mses_settings: dict, stencil: typing.List[dict], idx: int):
    """
    Updates the MSES settings dictionary from a given multipoint stencil and multipoint index

    Parameters
    ==========
    mses_settings: dict
      MSES settings dictionary

    stencil: typing.List[dict]
      A list of dictionaries describing the multipoint stencil, where each entry in the list is a dictionary
      representing a different stencil variable (Mach number, lift coefficient, etc.) and contains values for the
      variable name, index (used only in the case of transition location and actuator disk variables), and stencil
      point values (e.g., ``stencil_var["points"]`` may look something like ``[0.65, 0.70, 0.75]`` for Mach number)

    idx: int
      Index within the multipoint stencil used to update the MSES settings dictionary

    Returns
    =======
    dict
      The modified MSES settings dictionary
    """
    for stencil_var in stencil:
        if isinstance(mses_settings[stencil_var['variable']], list):
            mses_settings[stencil_var['variable']][stencil_var['index']] = stencil_var['points'][idx]
        else:
            mses_settings[stencil_var['variable']] = stencil_var['points'][idx]
    return mses_settings


def calculate_aero_data(airfoil_coord_dir: str, airfoil_name: str, coords: typing.Tuple[tuple],
                        tool: str = 'XFOIL', xfoil_settings: dict = None, mset_settings: dict = None,
                        mses_settings: dict = None, mplot_settings: dict = None, export_Cp: bool = True):
    r"""
    Convenience function calling either XFOIL or MSES depending on the ``tool`` specified

    Parameters
    ==========
    airfoil_coord_dir: str
      The directory containing the airfoil coordinate file

    airfoil_name: str
      A string describing the airfoil

    coords: typing.Tuple[tuple]
      If using XFOIL: specify a 2-D nested tuple array of size :math:`N \times 2`, where :math:`N` is the number of
      airfoil coordinates and the columns represent :math:`x` and :math:`y`. If using MSES, specify a 3-D nested
      tuple array of size :math:`M \times N \times 2`, where :math:`M` is the number of airfoils.

    tool: str
      The airfoil flow analysis tool to be used. Must be either ``"XFOIL"`` or ``"MSES"``. Default: ``"XFOIL"``

    xfoil_settings: dict
      A dictionary containing the settings for XFOIL. Must be specified if the ``"XFOIL"`` tool is selected.
      Default: ``None``

    mset_settings: dict
      A dictionary containing the settings for MSET. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    mses_settings: dict
      A dictionary containing the settings for MSES. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    mplot_settings: dict
      A dictionary containing the settings for MPLOT. Must be specified if the ``"MSES"`` tool is selected. Default:
      ``None``

    export_Cp: bool
      Whether to calculate and export the surface pressure coefficient distribution in the case of XFOIL, or the
      entire set of boundary layer data in the case of MSES. Default: ``True``

    export_CPK: bool
      Whether to calculate the mechanical flow power coefficient for the aero-propulsive case. Default: ``False``
    """

    tool_list = ['XFOIL', 'MSES']
    if tool not in tool_list:
        raise ValueError(f"\'tool\' must be one of {tool_list}")
    # if tool == 'XFOIL':
    #
    #     # Check for self-intersection and early return if self-intersecting:
    #     if check_airfoil_self_intersection(coords):
    #         return False

    aero_data = {}

    # Make the analysis directory if not already created
    base_dir = os.path.join(airfoil_coord_dir, airfoil_name)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if tool == 'XFOIL':
        if xfoil_settings is None:
            raise ValueError(f"\'xfoil_settings\' must be set if \'xfoil\' tool is selected")
        if 'xtr' not in xfoil_settings.keys():
            xfoil_settings['xtr'] = [1.0, 1.0]
        if 'N' not in xfoil_settings.keys():
            xfoil_settings['N'] = 9.0
        f = os.path.join(base_dir, airfoil_name + ".dat")
        if np.ndim(coords) == 2:
            write_tuple_tuple_to_file(f, coords)
        elif np.ndim(coords) == 3:
            write_tuple_tuple_to_file(f, coords[0])
        else:
            raise ValueError("Found coordinate set with dimension other than 2 or 3")
        xfoil_input_file = os.path.join(base_dir, 'xfoil_input.txt')
        xfoil_input_list = ['', 'oper', f'iter {xfoil_settings["iter"]}', 'visc', str(xfoil_settings['Re']),
                            f'M {xfoil_settings["Ma"]}',
                            'vpar', f'xtr {xfoil_settings["xtr"][0]} {xfoil_settings["xtr"][1]}',
                            f'N {xfoil_settings["N"]}', '']

        # alpha/Cl input setup (must choose exactly one of alpha, Cl, or CLI)
        if len([0 for prescribed_xfoil_val in (
                'alfa', 'Cl', 'CLI') if prescribed_xfoil_val in xfoil_settings.keys()]) != 1:
            raise ValueError('Either none or more than one of alpha, Cl, or CLI was set. '
                             'Choose exactly one for XFOIL analysis.')
        alpha = None
        Cl = None
        CLI = None
        if 'alfa' in xfoil_settings.keys():
            alpha = xfoil_settings['alfa']
        elif 'Cl' in xfoil_settings.keys():
            Cl = xfoil_settings['Cl']
        elif 'CLI' in xfoil_settings.keys():
            CLI = xfoil_settings['CLI']
        if alpha is not None:
            if not isinstance(alpha, list):
                alpha = [alpha]
            for idx, alf in enumerate(alpha):
                xfoil_input_list.append('alfa ' + str(alf))
        elif Cl is not None:
            if not isinstance(Cl, list):
                Cl = [Cl]
            for idx, Cl_ in enumerate(Cl):
                xfoil_input_list.append('Cl ' + str(Cl_))
        elif CLI is not None:
            if not isinstance(CLI, list):
                CLI = [CLI]
            for idx, CLI_ in enumerate(CLI):
                xfoil_input_list.append('CLI ' + str(CLI_))
        else:
            raise ValueError('At least one of alpha, Cl, or CLI must be set for XFOIL analysis.')

        if export_Cp:
            xfoil_input_list.append('cpwr ' + f"{airfoil_name}_Cp.dat")
        xfoil_input_list.append('')
        xfoil_input_list.append('quit')
        write_input_file(xfoil_input_file, xfoil_input_list)
        xfoil_log = os.path.join(base_dir, 'xfoil.log')
        # with open(xfoil_log, 'wb') as h:
        with open(xfoil_input_file, 'r') as g:
            process = subprocess.Popen(['xfoil', f"{airfoil_name}.dat"], stdin=g, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=base_dir, shell=False)
            aero_data['converged'] = False
            aero_data['timed_out'] = False
            aero_data['errored_out'] = False
            try:
                # print(f"communicating")
                outs, errs = process.communicate(timeout=xfoil_settings['timeout'])
                # print(f"done communicating")
                with open(xfoil_log, 'wb') as h:
                    h.write('Output:\n'.encode('utf-8'))
                    h.write(outs)
                    h.write('\nErrors:\n'.encode('utf-8'))
                    h.write(errs)
                aero_data['timed_out'] = False
                aero_data['converged'] = True
            except subprocess.TimeoutExpired:
                process.kill()
                outs, errs = process.communicate()
                with open(xfoil_log, 'wb') as h:
                    h.write('After timeout, \nOutput: \n'.encode('utf-8'))
                    h.write(outs)
                    h.write('\nErrors:\n'.encode('utf-8'))
                    h.write(errs)
                aero_data['timed_out'] = True
                aero_data['converged'] = False
            finally:
                if not aero_data['timed_out']:
                    # time.sleep(3)
                    line1, line2 = read_aero_data_from_xfoil(xfoil_log, aero_data)
                    if line1 is not None:
                        convert_xfoil_string_to_aero_data(line1, line2, aero_data)
                        if export_Cp:
                            aero_data['Cp'] = read_Cp_from_file_xfoil(os.path.join(base_dir, f"{airfoil_name}_Cp.dat"))

        # print(f"{aero_data = }")

        return aero_data, xfoil_log

    elif tool in ['mses', 'Mses', 'MSES']:
        aero_data['converged'] = False
        aero_data['timed_out'] = False
        aero_data['errored_out'] = False
        if mset_settings is None:
            raise ValueError(f"\'mset_settings\' must be set if \'mses\' tool is selected")
        if mses_settings is None:
            raise ValueError(f"\'mses_settings\' must be set if \'mses\' tool is selected")
        if mplot_settings is None:
            raise ValueError(f"\'mplot_settings\' must be set if \'mses\' tool is selected")

        converged = False
        mses_log, mplot_log = None, None
        mset_success, mset_log = run_mset(airfoil_name, airfoil_coord_dir, mset_settings, coords)

        # Set up single-point or multipoint settings
        mset_mplot_loop_iterations = 1
        stencil = None
        aero_data_list = None
        if 'multi_point_stencil' in mses_settings.keys() and mses_settings['multi_point_stencil'] is not None:
            stencil = mses_settings['multi_point_stencil']
            mset_mplot_loop_iterations = len(stencil[0]['points'])
            aero_data_list = []

        # Multipoint Loop
        for i in range(mset_mplot_loop_iterations):
            t1 = time.time()

            if stencil is not None:
                mses_settings = update_mses_settings_from_stencil(mses_settings=mses_settings, stencil=stencil, idx=i)
                # print(f"{mses_settings['XCDELH'] = }, {mses_settings['CLIFIN'] = }, {mses_settings['PTRHIN'] = }")

            if mset_success:
                converged, mses_log = run_mses(airfoil_name, airfoil_coord_dir, mses_settings)
            if mset_success and converged:
                mplot_log = run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='forces')
                aero_data = read_forces_from_mses(mplot_log)
                if export_Cp:
                    run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='Cp')
                    aero_data['BL'] = []
                    bl = read_bl_data_from_mses(os.path.join(airfoil_coord_dir, airfoil_name,
                                                             f"bl.{airfoil_name}"))
                    for side in range(len(bl)):
                        aero_data['BL'].append({})
                        for output_var in ['x', 'y', 'Cp']:
                            aero_data['BL'][-1][output_var] = bl[side][output_var]

                if mplot_settings["CPK"]:
                    mplot_settings["flow_field"] = 2
                    mplot_settings["Streamline_Grid"] = 2

                for mplot_output_name in ['Mach', 'Streamline_Grid', 'Grid', 'Grid_Zoom', 'flow_field']:
                    if mplot_settings[mplot_output_name]:
                        run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode=mplot_output_name)
                        if mplot_output_name == 'flow_field':
                            run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='grid_stats')

                if mplot_settings["CPK"]:
                    try:
                        CPK = calculate_CPK_mses(os.path.join(airfoil_coord_dir, airfoil_name))
                    except:
                        CPK = 1E9
                    aero_data["CPK"] = CPK

            t2 = time.time()
            # print(f"Time for stencil point {i}: {t2 - t1} seconds")

            if converged:
                aero_data['converged'] = True
                aero_data['timed_out'] = False
                aero_data['errored_out'] = False
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
            else:
                aero_data['converged'] = False
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
                break

            # print(f"{aero_data = }")

        logs = {'mset': mset_log, 'mses': mses_log, 'mplot': mplot_log}

        if aero_data_list is not None:
            aero_data = {k: [] for k in aero_data_list[0].keys()}
            for aero_data_set in aero_data_list:
                for k, v in aero_data_set.items():
                    aero_data[k].append(v)

        return aero_data, logs


def run_mset(name: str, base_dir: str, mset_settings: dict, coords: typing.Tuple[tuple]):
    r"""
    A Python API for MSET

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_dir: str
      MSET files will be stored in ``base_dir/name``

    mset_settings: dict
      Parameter set (dictionary)

    coords: typing.Tuple[tuple]
      A 3-D set of coordinates to write as the airfoil geometry. The array of coordinates has size
      :math:`M \times N \times 2` where :math:`M` is the number of airfoils and :math:`N` is the number of airfoil
      coordinates. The coordinates can be input as a ragged array, where :math:`N` changes with each 3-D slice (i.e.,
      the number of airfoil coordinates can be different for each airfoil).

    Returns
    =======
    bool, str
      A boolean describing whether the MSET call succeeded and a string containing the path to the MSET log file
    """
    write_blade_file(name, base_dir, mset_settings['grid_bounds'], coords)
    write_gridpar_file(name, base_dir, mset_settings)
    mset_input_name = 'mset_input.txt'
    mset_input_file = os.path.join(base_dir, name, mset_input_name)
    mset_input_list = ['1', '0', '2', '', '', '', '3', '4', '0']
    write_input_file(mset_input_file, mset_input_list)
    mset_log = os.path.join(base_dir, name, 'mset.log')
    with open(mset_log, 'wb') as f:
        with open(mset_input_file, 'r') as g:
            process = subprocess.Popen(['mset', name], stdin=g, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=os.path.join(base_dir, name), shell=False)
            try:
                outs, errs = process.communicate(timeout=mset_settings['timeout'])
                f.write('Output:\n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                mset_success = True
            except subprocess.TimeoutExpired:
                process.kill()
                outs, errs = process.communicate()
                f.write('After timeout, \nOutput: \n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                mset_success = False
    return mset_success, mset_log


# noinspection PyTypeChecker
def run_mses(name: str, base_folder: str, mses_settings: dict, stencil: bool = False):
    r"""
    A Python API for MSES

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_folder: str
      MSES files will be stored in ``base_folder/name``

    mses_settings: dict
      Flow parameter set (dictionary)

    stencil: bool
      Whether a multipoint stencil is to be used. This variable is only used here to determine whether to overwrite or
      append to the log file. Default: ``False``

    Returns
    =======
    bool, str
      A boolean describing whether the MSES solution is converged and a string containing the path to the MSES log file
    """
    write_mses_file(name, base_folder, mses_settings)
    mses_log = os.path.join(base_folder, name, 'mses.log')
    if stencil:
        read_write = 'ab'
    else:
        read_write = 'wb'
    converged = False
    with open(mses_log, read_write) as f:
        process = subprocess.Popen(['mses', name, str(mses_settings['iter'])], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=os.path.join(base_folder, name))
        try:
            outs, errs = process.communicate(timeout=mses_settings['timeout'])
            if 'Converged' in str(outs):
                converged = True
                if mses_settings['verbose']:
                    print('Converged!')
            else:
                if mses_settings['verbose']:
                    print('Not converged!')
            f.write('Output:\n'.encode('utf-8'))
            f.write(outs)
            f.write('\nErrors:\n'.encode('utf-8'))
            f.write(errs)
        except subprocess.TimeoutExpired:
            process.kill()
            outs, errs = process.communicate()
            f.write('After timeout, \nOutput: \n'.encode('utf-8'))
            f.write(outs)
            f.write('\nErrors:\n'.encode('utf-8'))
            f.write(errs)

    return converged, mses_log


def run_mplot(name: str, base_dir: str, mplot_settings: dict, mode: str = "forces", min_contour: float = 0.0,
              max_contour: float = 1.5, n_intervals: int = 0):
    r"""
    A Python API for MPLOT

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_dir: str
      MSES files will be stored in ``base_folder/name``

    mplot_settings: dict
      Flow parameter set (dictionary)

    mode: str
      What type of data to output from MPLOT. Current choices are ``"forces"``, ``"Cp"``, ``"flowfield"``,
      ``"grid_zoom"``, ``"grid"``, ``"grid_stats"``, and ``"Mach contours"``. Default: ``"forces"``

    min_contour: float
      Minimum contour level (only affects the result if ``mode=="Mach contours"``). Default: ``0.0``

    max_contour: float
      Maximum contour level (only affects the result if ``mode=="Mach contours"``). Default: ``1.5``

    n_intervals: int
      Number of contour levels (only affects the result if ``mode=="Mach contours"``). A value of ``0`` results in
      MPLOT automatically setting a "nice" value for the number of contour levels. Default: ``0``

    Returns
    =======
    str
      A string containing the path to the MPLOT log file
    """
    if mode in ["forces", "Forces"]:
        mplot_input_name = "mplot_forces_dump.txt"
        mplot_input_list = ['1', '12', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot.log')
    elif mode in ["CP", "cp", "Cp", "cP"]:
        mplot_input_name = "mplot_input_dumpcp.txt"
        mplot_input_list = ['12', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_cp.log')
    elif mode in ['flowfield', 'Flowfield', 'FlowField', 'flow_field']:
        mplot_input_name = "mplot_dump_flowfield.txt"
        mplot_input_list = ['11', '', 'Y', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_flowfield.log')
    elif mode in ['grid_zoom', 'Grid_Zoom', 'GRID_ZOOM']:
        mplot_input_name = "mplot_input_grid_zoom.txt"
        mplot_input_list = ['3', '1', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '1',
                            '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid_zoom.log')
    elif mode in ['grid', 'Grid', 'GRID']:
        mplot_input_name = "mplot_input_grid.txt"
        mplot_input_list = ['3', '1', '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid.log')
    elif mode in ['grid_stats', 'GridStats', 'Grid_Stats']:
        mplot_input_name = "mplot_input_grid_stats.txt"
        mplot_input_list = ['3', '10', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_grid_stats.log')
    elif mode in ["Streamline_Grid"]:
        mplot_input_name = "mplot_streamline_grid.txt"
        mplot_input_list = ['10', '', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_streamline_grid.log')
    elif mode in ["Mach", "mach", "M", "m", "Mach contours", "Mach Contours", "mach contours"]:
        mplot_input_name = "mplot_inputMachContours.txt"
        if n_intervals == 0:
            mplot_input_list = ['3', '3', 'M', '', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '3', 'M',
                                '', '8', '', '0']
        else:
            mplot_input_list = ['3', '3', 'M', '', '9', 'B', '', '6', '-0.441', '0.722', '1.937', '-0.626', '3', 'M',
                                f'{min_contour} {max_contour} {n_intervals}', '8', '', '0']
        mplot_log = os.path.join(base_dir, name, 'mplot_mach.log')
    else:
        raise Exception("Invalid MPLOT mode!")
    mplot_input_file = os.path.join(base_dir, name, mplot_input_name)
    write_input_file(mplot_input_file, mplot_input_list)
    with open(mplot_log, 'wb') as f:
        with open(mplot_input_file, 'r') as g:
            process = subprocess.Popen(['mplot', name], stdin=g, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=os.path.join(base_dir, name))
        try:
            outs, errs = process.communicate(timeout=mplot_settings['timeout'])
            f.write('Output:\n'.encode('utf-8'))
            f.write(outs)
            f.write('\nErrors:\n'.encode('utf-8'))
            f.write(errs)
        except subprocess.TimeoutExpired:
            process.kill()
            outs, errs = process.communicate()
            f.write('After timeout, \nOutput: \n'.encode('utf-8'))
            f.write(outs)
            f.write('\nErrors:\n'.encode('utf-8'))
            f.write(errs)
    if mode in ["Mach", "mach", "M", "m", "Mach contours", "Mach Contours", "mach contours"]:
        convert_ps_to_svg(os.path.join(base_dir, name),
                          'plot.ps',
                          'Mach_contours.pdf',
                          'Mach_contours.svg')
    elif mode in ['grid', 'Grid', 'GRID']:
        convert_ps_to_svg(os.path.join(base_dir, name), 'plot.ps', 'grid.pdf', 'grid.svg')
    elif mode in ['grid_zoom', 'Grid_Zoom', 'GRID_ZOOM']:
        convert_ps_to_svg(os.path.join(base_dir, name), 'plot.ps', 'grid_zoom.pdf', 'grid_zoom.svg')
    return mplot_log


def write_blade_file(name: str, base_dir: str, grid_bounds: typing.Iterable, coords: typing.Tuple[tuple]):
    r"""
    Writes airfoil geometry to an MSES blade file

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_dir: str
      Blade file will be stored as ``base_dir/name/blade.name``

    grid_bounds: typing.Iterable
      Iterable object containing the far-field boundary locations for MSES, of the form
      ``[x_lower, x_upper, y_lower, y_upper]``. For example, ``[-6, 6, -5, 5]`` will produce a pseudo-rectangular
      far-field boundary with :math:`x` going from -6 to 6 and :math:`y` going from -5 to 5. The boundary is not
      perfectly rectangular because MSES produces far-field boundaries that follow the streamlines close to the
      specified :math:`x` and :math:`y`-locations for the grid.

    coords: typing.Tuple[tuple]
      A 3-D set of coordinates to write as the airfoil geometry. The array of coordinates has size
      :math:`M \times N \times 2` where :math:`M` is the number of airfoils and :math:`N` is the number of airfoil
      coordinates. The coordinates can be input as a ragged array, where :math:`N` changes with each 3-D slice (i.e.,
      the number of airfoil coordinates can be different for each airfoil).
    """
    if not os.path.exists(os.path.join(base_dir, name)):  # if specified directory doesn't exist,
        os.mkdir(os.path.join(base_dir, name))  # create it
    blade_file = os.path.join(base_dir, name, 'blade.' + name)  # blade file stored as
    # <base_dir>\<name>\blade.<name>
    with open(blade_file, 'w') as f:  # Open the blade_file with write permission
        f.write(name + '\n')  # Write the name of the airfoil on the first line
        for gb in grid_bounds[0:-1]:  # For each of the entries in grid_bounds except the last (type: list),
            f.write(str(gb) + " ")  # write the grid_bound at the specified index with a space after it in blade_file
        f.write(str(grid_bounds[-1]) + "\n")
        for idx, airfoil_coords in enumerate(coords):
            for xy in airfoil_coords:
                f.write(f"{str(xy[0])} {str(xy[1])}\n")
            if idx != len(coords) - 1:
                f.write('999.0 999.0\n')  # Write 999.0 999.0 with a carriage return to indicate to MSES that another
                # airfoil follows
    return blade_file


def write_gridpar_file(name: str, base_folder: str, mset_settings: dict):
    """
    Writes grid parameters to a file readable by MSES

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_folder: str
      Grid parameter file will be stored as ``base_folder/name/gridpar.name``

    mset_settings: dict
      Parameter set (dictionary)

    Returns
    =======
    str
      Path of the created grid parameter file
    """
    if not os.path.exists(os.path.join(base_folder, name)):  # if specified directory doesn't exist,
        os.mkdir(os.path.join(base_folder, name))  # create it
    gridpar_file = os.path.join(base_folder, name, 'gridpar.' + name)
    with open(gridpar_file, 'w') as f:  # Open the gridpar_file with write permission
        f.write(f"{int(mset_settings['airfoil_side_points'])}\n")
        f.write(f"{mset_settings['exp_side_points']}\n")
        f.write(f"{int(mset_settings['inlet_pts_left_stream'])}\n")
        f.write(f"{int(mset_settings['outlet_pts_right_stream'])}\n")
        f.write(f"{int(mset_settings['num_streams_top'])}\n")
        f.write(f"{int(mset_settings['num_streams_bot'])}\n")
        f.write(f"{int(mset_settings['max_streams_between'])}\n")
        f.write(f"{mset_settings['elliptic_param']}\n")
        f.write(f"{mset_settings['stag_pt_aspect_ratio']}\n")
        f.write(f"{mset_settings['x_spacing_param']}\n")
        f.write(f"{mset_settings['alf0_stream_gen']}\n")

        multi_airfoil_grid = mset_settings['multi_airfoil_grid']

        for a in mset_settings['airfoil_order']:
            f.write(f"{multi_airfoil_grid[a]['dsLE_dsAvg']} {multi_airfoil_grid[a]['dsTE_dsAvg']} "
                    f"{multi_airfoil_grid[a]['curvature_exp']}\n")

        for a in mset_settings['airfoil_order']:
            f.write(f"{multi_airfoil_grid[a]['U_s_smax_min']} {multi_airfoil_grid[a]['U_s_smax_max']} "
                    f"{multi_airfoil_grid[a]['L_s_smax_min']} {multi_airfoil_grid[a]['L_s_smax_max']} "
                    f"{multi_airfoil_grid[a]['U_local_avg_spac_ratio']} {multi_airfoil_grid[a]['L_local_avg_spac_ratio']}\n")

    return gridpar_file


def write_mses_file(name: str, base_folder: str, mses_settings: dict):
    """
    Writes MSES flow parameters to a file

    Parameters
    ==========
    name: str
      Name of the airfoil [system]

    base_folder: str
      MSES flow parameter file will be stored as ``base_folder/name/mses.name``

    mses_settings: dict
      Parameter set (dictionary)

    Returns
    =======
    str
      Path of the created MSES flow parameter file
    """
    F = deepcopy(mses_settings)
    if not os.path.exists(os.path.join(base_folder, name)):  # if specified directory doesn't exist,
        os.mkdir(os.path.join(base_folder, name))  # create it
    mses_file = os.path.join(base_folder, name, 'mses.' + name)

    # ============= Reynolds number calculation =====================
    if not bool(F['viscous_flag']):
        F['REYNIN'] = 0.0

    if F['inverse_side'] % 2 != 0:
        F['ISMOVE'] = 1
        F['ISPRES'] = 1
    elif F['inverse_side'] == 0:
        F['ISMOVE'] = 0
        F['ISPRES'] = 0
    else:
        F['ISMOVE'] = 2
        F['ISPRES'] = 2

    with open(mses_file, 'w') as f:
        if F['target'] == 'alfa':
            global_constraint_target = 5  # Tell MSES to specify the angle of attack in degrees given by 'ALFIN'
        elif F['target'] == 'Cl':
            global_constraint_target = 6  # Tell MSES to target the lift coefficient specified by 'CLIFIN'
        else:
            raise ValueError('Invalid value for \'target\' (must be either \'alfa\' or \'Cl\')')

        if not F['inverse_flag']:
            f.write(f'3 4 5 7\n3 4 {global_constraint_target} 7\n')
        else:
            f.write(f'3 4 5 7 11 12\n3 4 {global_constraint_target} 7 11 12\n')

        f.write(f"{F['MACHIN']} {F['CLIFIN']} {F['ALFAIN']}\n")
        f.write(f"{int(F['ISMOM'])} {int(F['IFFBC'])}\n")
        f.write(f"{F['REYNIN']} {F['ACRIT']}\n")

        for idx in range(F['n_airfoils']):
            f.write(f"{F['XTRSupper'][idx]} {F['XTRSlower'][idx]}")
            if idx == F['n_airfoils'] - 1:
                f.write('\n')
            else:
                f.write(' ')

        f.write(f"{F['MCRIT']} {F['MUCON']}\n")

        if any([bool(flag) for flag in F['AD_flags']]) or bool(F['inverse_flag']):
            f.write(f"{int(F['ISMOVE'])} {int(F['ISPRES'])}\n")
            f.write(f"{int(F['NMODN'])} {int(F['NPOSN'])}\n")

        for idx, flag in enumerate(F['AD_flags']):
            if bool(flag):
                f.write(f"{int(F['ISDELH'][idx])} {F['XCDELH'][idx]} {F['PTRHIN'][idx]} {F['ETAH'][idx]}\n")

    mses_settings = F

    return mses_file, mses_settings


def write_input_file(input_file: str, input_list: typing.List[str]):
    """
    Writes inputs from a list to a file for use as STDIN commands to the shell/terminal.

    Parameters
    ==========
    input_file: str
      File where inputs are written

    input_list: typing.List[str]
      List of inputs to write. For example, passing ``["1", "", "12", "13"]`` is equivalent to typing the command
      sequence ``1, RETURN, RETURN, 12, RETURN, 13, RETURN`` into the shell or terminal.
    """
    with open(input_file, 'w') as f:
        for input_ in input_list:
            f.write(input_)
            f.write('\n')


def convert_xfoil_string_to_aero_data(line1: str, line2: str, aero_data: dict):
    """
    Extracts aerodynamic data from strings pulled from XFOIL log files. The two string inputs are the string outputs
    from ``pymead.analysis.read_aero_data.read_aero_data_from_xfoil``

    Parameters
    ==========
    line1: str
      First line containing the aerodynamic data in the XFOIL log file.

    line2: str
      Second line containing the aerodynamic data in the XFOIL log file.

    aero_data: dict
      Dictionary to which to write the aerodynamic data
    """
    new_str = line1.replace(' ', '') + line2.replace(' ', '')
    new_str = new_str.replace('=>', '')
    appending = False
    data_list = []
    for ch in new_str:
        if ch.isdigit() or ch == '.' or ch == '-':
            if appending:
                data_list[-1] += ch
        else:
            appending = False
        last_ch = ch
        if last_ch == '=' and not ch == '>':
            appending = True
            data_list.append('')
    aero_data['alf'] = float(data_list[4])
    aero_data['Cm'] = float(data_list[0])
    aero_data['Cd'] = float(data_list[1])
    aero_data['Cdf'] = float(data_list[2])
    aero_data['Cdp'] = float(data_list[3])
    aero_data['Cl'] = float(data_list[5])
    aero_data['L/D'] = aero_data['Cl'] / aero_data['Cd']
    return aero_data


def convert_cell_centered_to_edge_centered(grid_shape: np.ndarray, flow_var_cell_centered: np.ndarray):
    flow_var_edge_centered = np.zeros(grid_shape)
    i_max = flow_var_edge_centered.shape[0]
    j_max = flow_var_edge_centered.shape[1]

    import time

    t1 = time.time()

    for i in range(i_max):
        for j in range(j_max):
            # Corner cases
            if i == 0 and j == 0:
                flow_var_edge_centered[i, j] = flow_var_cell_centered[i, j]
            elif i == 0 and j == j_max - 1:
                flow_var_edge_centered[i, j] = flow_var_cell_centered[i, j - 1]
            elif i == i_max - 1 and j == 0:
                flow_var_edge_centered[i, j] = flow_var_cell_centered[i - 1, j]
            elif i == i_max - 1 and j == j_max - 1:
                flow_var_edge_centered[i, j] = flow_var_cell_centered[i - 1, j - 1]

            # Edge cases
            elif i == 0:
                flow_var_edge_centered[i, j] = np.mean([flow_var_cell_centered[i, j],
                                                        flow_var_cell_centered[i, j - 1]])
            elif i == i_max - 1:
                flow_var_edge_centered[i, j] = np.mean([flow_var_cell_centered[i - 1, j],
                                                        flow_var_cell_centered[i - 1, j - 1]])
            elif j == 0:
                flow_var_edge_centered[i, j] = np.mean([flow_var_cell_centered[i, j],
                                                        flow_var_cell_centered[i - 1, j]])
            elif j == j_max - 1:
                flow_var_edge_centered[i, j] = np.mean([flow_var_cell_centered[i, j - 1],
                                                        flow_var_cell_centered[i - 1, j - 1]])

            # Normal case
            else:
                flow_var_edge_centered[i, j] = np.mean([flow_var_cell_centered[i, j],
                                                        flow_var_cell_centered[i - 1, j],
                                                        flow_var_cell_centered[i, j - 1],
                                                        flow_var_cell_centered[i - 1, j - 1]
                                                        ])

    t2 = time.time()
    # print(f"Elapsed time = {t2 - t1} seconds")

    return flow_var_edge_centered


def extrapolate_data_line_mses_field(flow_vars_edge_centered: typing.List[np.ndarray],
                                     x_grid_section: np.ndarray, y_grid_section: np.ndarray, point: np.ndarray,
                                     angle: float, bl_data_lower: dict, bl_data_upper: dict,
                                     line_extend: float = 10.0, n_points: int = 30):
    lower_streamline = np.column_stack((x_grid_section[:, 0], y_grid_section[:, 0]))
    upper_streamline = np.column_stack((x_grid_section[:, -1], y_grid_section[:, -1]))
    lower_bl = np.column_stack((np.array(bl_data_lower["x"]), np.array(bl_data_lower["y"])))
    upper_bl = np.column_stack((np.array(bl_data_upper["x"]), np.array(bl_data_upper["y"])))
    lower_streamline_shapely = convert_numpy_array_to_shapely_LineString(lower_streamline)
    upper_streamline_shapely = convert_numpy_array_to_shapely_LineString(upper_streamline)
    lower_bl_shapely = convert_numpy_array_to_shapely_LineString(lower_bl)
    upper_bl_shapely = convert_numpy_array_to_shapely_LineString(upper_bl)
    upper_point = point + line_extend * np.array([np.cos(angle), np.sin(angle)])
    lower_point = point - line_extend * np.array([np.cos(angle), np.sin(angle)])
    long_line = np.row_stack((upper_point, lower_point))
    long_line_shapely = convert_numpy_array_to_shapely_LineString(long_line)
    upper_inters = long_line_shapely.intersection(upper_streamline_shapely)
    lower_inters = long_line_shapely.intersection(lower_streamline_shapely)
    upper_bl_inters = long_line_shapely.intersection(upper_bl_shapely)
    lower_bl_inters = long_line_shapely.intersection(lower_bl_shapely)
    if isinstance(upper_inters, MultiPoint):
        upper_inters = upper_inters.geoms[0]
    if isinstance(lower_inters, MultiPoint):
        lower_inters = lower_inters.geoms[0]
    if isinstance(upper_bl_inters, MultiPoint):
        upper_bl_inters = upper_bl_inters.geoms[0]
    if isinstance(lower_bl_inters, MultiPoint):
        lower_bl_inters = lower_bl_inters.geoms[0]

    # import matplotlib.pyplot as plt
    # plt.plot(lower_bl[:, 0], lower_bl[:, 1], color="red")
    # plt.plot(upper_bl[:, 0], upper_bl[:, 1], color="green")
    # plt.plot(long_line[:, 0], long_line[:, 1], color="blue")
    # plt.show()

    def evaluate_bl_at_point(bl_data_dict: dict, x_point: float):
        x_array = np.array(bl_data_dict["x"])
        Cp_array = np.array(bl_data_dict["Cp"])
        ue_array = np.array(bl_data_dict["Ue/Uinf"])
        rhoe_array = np.array(bl_data_dict["rhoe/rhoinf"])
        deltastar_array = np.array(bl_data_dict["delta*"])
        thetastar_array = np.array(bl_data_dict["theta*"])
        bl_at_point = {
            "x": x_point,
            "Cp": np.interp(x_point, x_array, Cp_array),
            "ue": np.interp(x_point, x_array, ue_array),
            "rhoe": np.interp(x_point, x_array, rhoe_array),
            "deltastar": np.interp(x_point, x_array, deltastar_array),
            "thetastar": np.interp(x_point, x_array, thetastar_array),
        }
        return bl_at_point

    # Approximate the boundary layer thickness by taking the distance from the intersection point on the airfoil surface
    # to the intersection point on the boundary layer edge along the extraction line (the true value would be taking it
    # as the perpendicular distance from the surface. For small surface angles, this is a good approximation.)
    if hasattr(lower_bl_inters, "x") and hasattr(lower_bl_inters, "y"):
        bl_at_point_lower = evaluate_bl_at_point(bl_data_lower, lower_bl_inters.x)
        bl_at_point_lower["delta"] = np.hypot(lower_inters.x - lower_bl_inters.x, lower_inters.y - lower_bl_inters.y)
    else:
        bl_at_point_lower = None

    if hasattr(upper_bl_inters, "x") and hasattr(upper_bl_inters, "y"):
        bl_at_point_upper = evaluate_bl_at_point(bl_data_upper, upper_bl_inters.x)
        bl_at_point_upper["delta"] = np.hypot(upper_inters.x - upper_bl_inters.x, upper_inters.y - upper_bl_inters.y)
    else:
        bl_at_point_upper = None

    line_to_extract = np.linspace(np.array([lower_inters.x, lower_inters.y]),
                                  np.array([upper_inters.x, upper_inters.y]), n_points)

    x_interp = x_grid_section.flatten()
    y_interp = y_grid_section.flatten()
    xy_interp = np.column_stack((x_interp, y_interp))
    output_value_list = []
    for flow_var in flow_vars_edge_centered:
        values = flow_var.flatten()
        interp = CloughTocher2DInterpolator(xy_interp, values)
        output_values = interp(line_to_extract)
        output_value_list.append(output_values)
    return np.column_stack((line_to_extract, *output_value_list)), bl_at_point_upper, bl_at_point_lower


def line_integral_CPK_inviscid(Cp: np.ndarray, rho: np.ndarray, u: np.ndarray,
                               v: np.ndarray, V: np.ndarray, x: np.ndarray, y: np.ndarray, n_hat_right: bool):
    """
    Computes the mechanical flow power line integral in the inviscid streamtube only according to
    Drela's "Power Balance in Aerodynamic Flows", a 2009 AIAA Journal article

    Parameters
    ==========
    Cp: np.ndarray
        Node-centered pressure coefficient field

    rho: np.ndarray
        Node-centered density field

    u: np.ndarray
        Node-centered x-velocity field

    v: np.ndarray
        Node-centered y-velocity field

    V: np.ndarray
        Node-centered velocity magnitude field

    x: np.ndarray
        1-d array of x-coordinates of the line along which to compute the line integral

    y: np.ndarray
        1-d array of y-coordinates of the line along which to compute the line integral

    n_hat_right: bool
        Whether the perpendicular n_hat vector points to the right of the line (If ``True``, computed as the sum of
        the arctan of the value of dx/dy of the line less 90 degrees. If ``False``, 90 degrees is added instead).
    """
    # Calculate the direction of n_hat
    dx_dy = np.gradient(y, x)
    angle = np.arctan2(dx_dy, 1)
    perp_angle = -np.pi / 2 if n_hat_right else np.pi / 2
    n_hat = np.column_stack((np.cos(angle + perp_angle), np.sin(angle + perp_angle)))
    V_vec = np.column_stack((u, v))

    # Compute the dot product V_vec * n_hat
    dot_product = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec, n_hat)])
    # integrand = (-Cp - rho * V**2 + rho) * dot_product.flatten()
    integrand = (rho * (1 - V**2) - Cp) * dot_product.flatten()

    # Build the length increment vector (dl)
    dl = np.array([0.0])
    dl = np.append(dl, np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]))  # compute incremental length along x and y
    # print(f"Before cumulative summation, {dl = }")
    # print(f"{x = }")
    # print(f"{y = }")
    dl = np.cumsum(dl)

    # Integrate
    integral = np.trapz(integrand, dl)
    return integral


def line_integral_CPK_bl(Cp, delta, ue, rhoe, deltastar, thetastar, outlet: bool):
    """
    Computes CPK at a given boundary layer location

    Parameters
    ==========
    Cp: float
        Pressure coefficient

    delta: float
        Boundary layer thickness (:math:`\delta / c_\text{main}`)

    ue: float
        Boundary layer edge velocity (:math:`u_e / V_\infty`)

    rhoe: float
        Boundary layer edge density (:math:`\rho_e / \rho_\infty`)

    deltastar: float
        Boundary layer displacement thickness (:math:`\delta^* / c_\text{main}`)

    thetastar: float
        Boundary layer kinetic energy thickness (:math:`\theta^* / c_\text{main}`)

    outlet: bool
        Whether this boundary layer is part of an outlet
    """
    # delta = 8 * deltastar
    # integral = (7/8) * delta * ue * Cp + rhoe * ue * (
    #         delta - deltastar) + rhoe * ue**3 * (delta - deltastar - thetastar)
    # print(f"{delta = }, {ue = }, {Cp = }, {rhoe = }, {deltastar = }, {thetastar = }, {integral = }")
    # return integral if outlet else -integral  # -V_vec dot n_hat is approximately V for the outlet and -V for the inlet
    return 0.0


def calculate_CPK_mses(analysis_subdir: str, configuration: str = "underwing_te"):
    """
    A specialized function that calculates the mechanical flow power coefficient for an underwing trailing edge
    aero-propulsive configuration.
    """
    if configuration != "underwing_te":
        raise NotImplementedError("Only the underwing trailing edge configuration is currently implemented")

    airfoil_system_name = os.path.split(analysis_subdir)[-1]
    field_file = os.path.join(analysis_subdir, f'field.{airfoil_system_name}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{airfoil_system_name}')
    blade_file = os.path.join(analysis_subdir, f"blade.{airfoil_system_name}")
    bl_file = os.path.join(analysis_subdir, f"bl.{airfoil_system_name}")
    coords = convert_blade_file_to_3d_array(blade_file)

    field = read_field_from_mses(field_file)
    bl_data = read_bl_data_from_mses(bl_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)

    nacelle_le = coords[2][150 + 2 * 149, :]
    nacelle_te = (coords[2][0, :] + coords[2][-1, :]) / 2
    dydx = (nacelle_te[1] - nacelle_le[1]) / (nacelle_te[0] - nacelle_le[0])
    nacelle_chord = np.hypot(nacelle_te[0] - nacelle_le[0], nacelle_te[1] - nacelle_le[1])
    chord_5perc = nacelle_le + 0.05 * nacelle_chord * np.array([1 / np.sqrt(dydx**2 + 1), dydx / np.sqrt(dydx**2 + 1)])

    main_te_lower = coords[0][-1, :]
    nacelle_te_upper = coords[2][0, :]

    angle = np.arctan2(main_te_lower[1] - nacelle_te_upper[1], main_te_lower[0] - nacelle_te_upper[0])

    plot_planes = False
    # PLOT INLET AND OUTLET PLANES
    if plot_planes:
        for c_set in coords:
            plt.plot(c_set[:, 0], c_set[:, 1])
        plt.plot(np.array([chord_5perc[0], chord_5perc[0] + 0.3 * np.cos(angle)]),
                 np.array([chord_5perc[1], chord_5perc[1] + 0.3 * np.sin(angle)]))
        plt.plot(np.array([nacelle_te[0], nacelle_te[0] + 0.3 * np.cos(angle)]),
                 np.array([nacelle_te[1], nacelle_te[1] + 0.3 * np.sin(angle)]))
        plt.gca().set_aspect("equal")

        plt.show()

    CPK = 0.0

    flow_sections = [1, 2]

    # inlet, outlet = {"p": [], "rho": [], "u": [], "v": []}, {"p": [], "rho": [], "u": [], "v": []}

    underwing_flow_section_bl_map = {
        1: (4, 3),
        2: (2, 1)
    }

    start_idx, end_idx = 0, x_grid[0].shape[1] - 1
    for flow_section_idx in range(grid_stats["numel"] + 1):
        if flow_section_idx in flow_sections:
            Cp = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                       field[flow_var_idx["Cp"]][:, start_idx:end_idx])
            rho = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                         field[flow_var_idx["rho"]][:, start_idx:end_idx])
            u = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                       field[flow_var_idx["u"]][:, start_idx:end_idx])
            v = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                       field[flow_var_idx["v"]][:, start_idx:end_idx])
            V = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                       field[flow_var_idx["q"]][:, start_idx:end_idx])

            bl_data_lower = bl_data[underwing_flow_section_bl_map[flow_section_idx][0]]
            bl_data_upper = bl_data[underwing_flow_section_bl_map[flow_section_idx][1]]

            xyCprhouvV_out, bl_at_point_upper_out, bl_at_point_lower_out = extrapolate_data_line_mses_field(
                [Cp, rho, u, v, V], x_grid[flow_section_idx], y_grid[flow_section_idx],
                bl_data_lower=bl_data_lower, bl_data_upper=bl_data_upper, point=nacelle_te_upper, angle=angle
            )

            xyCprhouvV_in, bl_at_point_upper_in, bl_at_point_lower_in = extrapolate_data_line_mses_field(
                [Cp, rho, u, v, V], x_grid[flow_section_idx], y_grid[flow_section_idx],
                bl_data_lower=bl_data_lower, bl_data_upper=bl_data_upper, point=chord_5perc, angle=angle
            )

            # Integrate over the propulsor outlet for the given flow section
            Cp_out, rho_out, u_out, v_out, V_out = \
                xyCprhouvV_out[:, 2], xyCprhouvV_out[:, 3], xyCprhouvV_out[:, 4], xyCprhouvV_out[:, 5], xyCprhouvV_out[:, 6]
            x_out = xyCprhouvV_out[:, 0]
            y_out = xyCprhouvV_out[:, 1]
            outlet_integral = line_integral_CPK_inviscid(Cp_out, rho_out, u_out, v_out, V_out, x_out, y_out,
                                                         n_hat_right=False)  # n_hat points into the propulsor
            CPK += outlet_integral
            # print(f"{outlet_integral = }")

            # if bl_at_point_upper_out is not None:
            #     CPK += line_integral_CPK_bl(bl_at_point_upper_out["Cp"], bl_at_point_upper_out["delta"],
            #                                 bl_at_point_upper_out["ue"], bl_at_point_upper_out["rhoe"],
            #                                 bl_at_point_upper_out["deltastar"], bl_at_point_upper_out["thetastar"],
            #                                 outlet=True)
            #
            # if bl_at_point_lower_out is not None:
            #     CPK += line_integral_CPK_bl(bl_at_point_lower_out["Cp"], bl_at_point_lower_out["delta"],
            #                                 bl_at_point_lower_out["ue"], bl_at_point_lower_out["rhoe"],
            #                                 bl_at_point_lower_out["deltastar"], bl_at_point_lower_out["thetastar"],
            #                                 outlet=True)

            # print(f"After adding the BL CPK, {CPK - outlet_integral = }")

            # Integrate over the propulsor inlet for the given flow section
            Cp_in, rho_in, u_in, v_in, V_in = \
                xyCprhouvV_in[:, 2], xyCprhouvV_in[:, 3], xyCprhouvV_in[:, 4], xyCprhouvV_in[:, 5], xyCprhouvV_in[:, 6]
            x_in = xyCprhouvV_in[:, 0]
            y_in = xyCprhouvV_in[:, 1]
            inlet_integral = line_integral_CPK_inviscid(Cp_in, rho_in, u_in, v_in, V_in, x_in, y_in,
                                                        n_hat_right=True)  # n_hat points into the propulsor
            CPK += inlet_integral
            # CPK_temp = CPK
            # print(f"{inlet_integral = }, {CPK = }")

            # if bl_at_point_upper_in is not None:
            #     CPK += line_integral_CPK_bl(bl_at_point_upper_in["Cp"], bl_at_point_upper_in["delta"],
            #                                 bl_at_point_upper_in["ue"], bl_at_point_upper_in["rhoe"],
            #                                 bl_at_point_upper_in["deltastar"], bl_at_point_upper_in["thetastar"],
            #                                 outlet=False)
            #
            # if bl_at_point_lower_in is not None:
            #     CPK += line_integral_CPK_bl(bl_at_point_lower_in["Cp"], bl_at_point_lower_in["delta"],
            #                                 bl_at_point_lower_in["ue"], bl_at_point_lower_in["rhoe"],
            #                                 bl_at_point_lower_in["deltastar"], bl_at_point_lower_in["thetastar"],
            #                                 outlet=False)

            # print(f"After adding the inlet BL contributions, {CPK - CPK_temp = }")

        if flow_section_idx < grid_stats["numel"]:
            start_idx = end_idx
            end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

    if np.isnan(CPK):
        CPK = 1e9

    # print(f"{CPK = }, {inlet_integral = }, {outlet_integral = }")

    return CPK


class GeometryError(Exception):
    pass
