import subprocess
import os
import typing
from copy import deepcopy
import time
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from shapely.geometry import MultiPoint

from pymead.core.mea import MEA
from pymead.analysis.read_aero_data import read_aero_data_from_xfoil, read_Cp_from_file_xfoil, read_bl_data_from_mses, \
    read_forces_from_mses, read_grid_stats_from_mses, read_field_from_mses, read_streamline_grid_from_mses, \
    flow_var_idx, convert_blade_file_to_3d_array, read_actuator_disk_data_mses, read_Mach_from_mses_file
from pymead.analysis.compressible_flow import calculate_normal_shock_total_pressure_ratio
from pymead.utils.file_conversion import convert_ps_to_svg
from pymead.utils.geometry import convert_numpy_array_to_shapely_LineString
from pymead.utils.read_write_files import save_data
import os
import subprocess
import time
import typing
from collections import namedtuple
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from shapely.geometry import MultiPoint

from pymead.analysis.compressible_flow import calculate_normal_shock_total_pressure_ratio
from pymead.analysis.read_aero_data import read_aero_data_from_xfoil, read_Cp_from_file_xfoil, read_bl_data_from_mses, \
    read_forces_from_mses, read_grid_stats_from_mses, read_field_from_mses, read_streamline_grid_from_mses, \
    flow_var_idx, convert_blade_file_to_3d_array, read_actuator_disk_data_mses, read_Mach_from_mses_file
from pymead.core.mea import MEA
from pymead.utils.file_conversion import convert_ps_to_svg
from pymead.utils.geometry import convert_numpy_array_to_shapely_LineString
from pymead.utils.read_write_files import save_data

SVG_PLOTS = ['Mach_contours', 'grid', 'grid_zoom']
SVG_SETTINGS_TR = {
    SVG_PLOTS[0]: 'Mach',
    SVG_PLOTS[1]: 'Grid',
    SVG_PLOTS[2]: 'Grid_Zoom',
}


def update_xfoil_settings_from_stencil(xfoil_settings: dict, stencil: typing.List[dict], idx: int):
    """
    Updates the XFOIL settings dictionary from a given multipoint stencil and multipoint index

    Parameters
    ==========
    xfoil_settings: dict
      MSES settings dictionary

    stencil: typing.List[dict]
      A list of dictionaries describing the multipoint stencil, where each entry in the list is a dictionary
      representing a different stencil variable (Mach number, lift coefficient, etc.) and contains values for the
      variable name, index (not used in XFOIL), and stencil
      point values (e.g., ``stencil_var["points"]`` may look something like ``[0.65, 0.70, 0.75]`` for Mach number)

    idx: int
      Index within the multipoint stencil used to update the XFOIL settings dictionary

    Returns
    =======
    dict
      The modified XFOIL settings dictionary
    """
    for stencil_var in stencil:
        if isinstance(xfoil_settings[stencil_var['variable']], list):
            xfoil_settings[stencil_var['variable']][stencil_var['index']] = stencil_var['points'][idx]
        else:
            xfoil_settings[stencil_var['variable']] = stencil_var['points'][idx]
    return xfoil_settings


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


def calculate_aero_data(airfoil_coord_dir: str, airfoil_name: str,
                        coords: typing.List[np.ndarray] = None,
                        mea: MEA = None, mea_airfoil_names: typing.List[str] = None,
                        tool: str = 'XFOIL', xfoil_settings: dict = None, mset_settings: dict = None,
                        mses_settings: dict = None, mplot_settings: dict = None, export_Cp: bool = True,
                        save_aero_data: bool = True):
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

    save_aero_data: bool
        Whether to save the aerodynamic data as a JSON file to the analysis directory

    Returns
    =======
    dict, str
        A dictionary containing the evaluated aerodynamic data and the path to the log file
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

        xfoil_log = None

        # Set up single-point or multipoint settings
        xfoil_loop_iterations = 1
        stencil = None
        aero_data_list = None
        if 'multi_point_stencil' in xfoil_settings.keys() and xfoil_settings['multi_point_stencil'] is not None:
            stencil = xfoil_settings['multi_point_stencil']
            xfoil_loop_iterations = len(stencil[0]['points'])
            aero_data_list = []

        # Multipoint Loop
        for i in range(xfoil_loop_iterations):

            if stencil is not None:
                xfoil_settings = update_xfoil_settings_from_stencil(xfoil_settings=xfoil_settings, stencil=stencil, idx=i)
                # print(f"{mses_settings['XCDELH'] = }, {mses_settings['CLIFIN'] = }, {mses_settings['PTRHIN'] = }")

            aero_data, xfoil_log = run_xfoil(airfoil_name, base_dir, xfoil_settings, coords, export_Cp=export_Cp)

            if aero_data["converged"]:
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
            else:
                if aero_data_list is not None:
                    aero_data_list.append(aero_data)
                break

        if aero_data_list is not None:
            aero_data = {k: [] for k in aero_data_list[0].keys()}
            for aero_data_set in aero_data_list:
                for k, v in aero_data_set.items():
                    aero_data[k].append(v)

        if save_aero_data:
            if aero_data["converged"]:
                for k, v in aero_data["Cp"].items():
                    if isinstance(v, np.ndarray):
                        aero_data["Cp"][k] = v.tolist()
            save_data(aero_data, os.path.join(base_dir, "aero_data.json"))

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
        mset_success, mset_log, airfoil_name_order = run_mset(
            airfoil_name, airfoil_coord_dir, mset_settings, coords=coords, mea=mea, mea_airfoil_names=mea_airfoil_names)

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
                converged, mses_log = run_mses(airfoil_name, airfoil_coord_dir, mses_settings,
                                               airfoil_name_order=airfoil_name_order)
            if mset_success and converged:
                mplot_log = run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='forces')
                aero_data = read_forces_from_mses(mplot_log)

                # This is error is triggered in read_aero_data() in the rare case that there is an error reading the
                # force file. If this error is triggered, break out of them multipoint loop.
                errored_out = np.isclose(aero_data["Cd"], 1000.0)
                if errored_out:
                    aero_data["converged"] = True
                    aero_data["timed_out"] = False
                    aero_data['errored_out'] = True
                    if aero_data_list is not None:
                        aero_data_list.append(aero_data)
                    break

                if export_Cp:
                    run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode="Cp")
                    aero_data["BL"] = []
                    for attempt in range(100):
                        if attempt > 0:
                            print(f"{attempt = }")
                        try:
                            bl = read_bl_data_from_mses(os.path.join(airfoil_coord_dir, airfoil_name,
                                                                     f"bl.{airfoil_name}"))
                            for side in range(len(bl)):
                                aero_data["BL"].append({})
                                for output_var in ["x", "y", "Cp"]:
                                    aero_data["BL"][-1][output_var] = bl[side][output_var]
                            break
                        except KeyError:
                            time.sleep(0.01)

                if mplot_settings["CPK"]:
                    mplot_settings["flow_field"] = 2
                    mplot_settings["Streamline_Grid"] = 2

                if mplot_settings["flow_field"]:
                    mplot_settings["Streamline_Grid"] = 2

                for mplot_output_name in ['Mach', 'Streamline_Grid', 'Grid', 'Grid_Zoom', 'flow_field']:
                    if mplot_settings[mplot_output_name]:
                        run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode=mplot_output_name)
                        if mplot_output_name == 'flow_field':
                            run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='grid_stats')

                if mplot_settings["CPK"]:
                    try:
                        # outputs_CPK = calculate_CPK_power_consumption(os.path.join(airfoil_coord_dir, airfoil_name))
                        # outputs_CPK = calculate_CPK_power_consumption_old(os.path.join(airfoil_coord_dir, airfoil_name))
                        outputs_CPK = calculate_CPK_mses_inviscid_only(os.path.join(airfoil_coord_dir, airfoil_name))
                    except Exception as e:
                        print(f"{e = }")
                        # outputs_CPK = {"CPK": 1e9, "diss_shock": 1e9, "diss_surf": 1e9, "Edot": 1e9, "Cd": 1e9,
                        #                "Edota": 1e9, "Edotp": 1e9}
                        outputs_CPK = {"CPK": 1e9}
                    aero_data = {**aero_data, **outputs_CPK}

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
                aero_data["timed_out"] = False
                aero_data["errored_out"] = False
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

        if save_aero_data:
            save_data(aero_data, os.path.join(airfoil_coord_dir, airfoil_name, "aero_data.json"))

        return aero_data, logs


def run_xfoil(airfoil_name: str, base_dir: str, xfoil_settings: dict, coords: np.ndarray,
              export_Cp: bool = True):
    aero_data = {}

    if "xtr" not in xfoil_settings.keys():
        xfoil_settings["xtr"] = [1.0, 1.0]
    if 'N' not in xfoil_settings.keys():
        xfoil_settings['N'] = 9.0
    f = os.path.join(base_dir, airfoil_name + ".dat")

    # Attempt to save the file
    save_attempts = 0
    max_save_attempts = 100
    while True:
        save_attempts += 1
        if save_attempts > max_save_attempts:
            raise ValueError("Exceeded the maximum number of allowed coordinate file save attempts")
        try:
            np.savetxt(f, coords)
            break
        except OSError:
            time.sleep(0.01)

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
            xfoil_input_list.append(f"alfa {alf}")
    elif Cl is not None:
        if not isinstance(Cl, list):
            Cl = [Cl]
        for idx, Cl_ in enumerate(Cl):
            xfoil_input_list.append(f"Cl {Cl_}")
    elif CLI is not None:
        if not isinstance(CLI, list):
            CLI = [CLI]
        for idx, CLI_ in enumerate(CLI):
            xfoil_input_list.append(f"CLI {CLI_}")
    else:
        raise ValueError('At least one of alpha, Cl, or CLI must be set for XFOIL analysis.')

    if export_Cp:
        xfoil_input_list.append('cpwr ' + f"{airfoil_name}_Cp.dat")
    xfoil_input_list.append('')
    xfoil_input_list.append('quit')
    write_input_file(xfoil_input_file, xfoil_input_list)
    xfoil_log = os.path.join(base_dir, 'xfoil.log')
    with open(xfoil_input_file, 'r') as g:
        process = subprocess.Popen(['xfoil', f"{airfoil_name}.dat"], stdin=g, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=base_dir, shell=False)
        aero_data['converged'] = False
        aero_data['timed_out'] = False
        aero_data['errored_out'] = False
        try:
            outs, errs = process.communicate(timeout=xfoil_settings['timeout'])
            with open(xfoil_log, 'wb') as h:
                h.write('Output:\n'.encode('utf-8'))
                h.write(outs)
                h.write('\nErrors:\n'.encode('utf-8'))
                h.write(errs)
            aero_data['timed_out'] = False
            aero_data['converged'] = True

            # This commented code is currently not specific enough to handle only global convergence failure
            # (and not catch local convergence failures)
            # with open(xfoil_log, "r") as log_file:
            #     for line in log_file:
            #         if "Convergence failed" in line:
            #             print(f"Convergence failed! {log_file = }")
            #             aero_data["converged"] = False
            #             break

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
            if not aero_data['timed_out'] and aero_data["converged"]:
                line1, line2 = read_aero_data_from_xfoil(xfoil_log, aero_data)
                if line1 is not None:
                    convert_xfoil_string_to_aero_data(line1, line2, aero_data)
                    if export_Cp:
                        aero_data['Cp'] = read_Cp_from_file_xfoil(os.path.join(base_dir, f"{airfoil_name}_Cp.dat"))

    return aero_data, xfoil_log


def run_mset(name: str, base_dir: str, mset_settings: dict, mea_airfoil_names: typing.List[str],
             coords: typing.List[np.ndarray] = None, mea: MEA = None):
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
    if coords is None and mea is None:
        raise ValueError("Must specify either coords or mea")
    if coords is not None and mea is not None:
        raise ValueError("Cannot specify both coords and mea")

    if coords is not None:
        blade_file_path, airfoil_name_order = write_blade_file(name, base_dir, mset_settings['grid_bounds'], coords,
                                                               mea_airfoil_names=mea_airfoil_names)
    elif mea is not None:
        blade_file_path, airfoil_name_order = mea.write_mses_blade_file(
            name, os.path.join(base_dir, name), grid_bounds=mset_settings["grid_bounds"],
            max_airfoil_points=mset_settings["downsampling_max_pts"] if bool(mset_settings["use_downsampling"]) else None,
            curvature_exp=mset_settings["downsampling_curve_exp"])
    else:
        raise ValueError("At least one of either coords or mea must be specified to write the blade file")

    write_gridpar_file(name, base_dir, mset_settings, airfoil_name_order=airfoil_name_order)
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
    return mset_success, mset_log, airfoil_name_order


def run_mses(name: str, base_folder: str, mses_settings: dict, airfoil_name_order: typing.List[str],
             stencil: bool = False):
    r"""
    A Python API for MSES

    Parameters
    ----------
    name: str
        Name of the airfoil [system]

    base_folder: str
        MSES files will be stored in ``base_folder/name``

    mses_settings: dict
        Flow parameter set (dictionary)

    airfoil_name_order: typing.List[str]
        List of the names of the airfoils (from top to bottom)

    stencil: bool
        Whether a multipoint stencil is to be used. This variable is only used here to determine whether to overwrite or
        append to the log file. Default: ``False``

    Returns
    -------
    bool, str
        A boolean describing whether the MSES solution is converged and a string containing the path to the MSES
        log file
    """
    write_mses_file(name, base_folder, mses_settings, airfoil_name_order=airfoil_name_order)
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

    mplot_attempts = 0
    mplot_max_attempts = 100
    while mplot_attempts < mplot_max_attempts:
        try:
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
            break
        except OSError:
            # In case any of the log files cannot be created/read temporarily, wait a short period of time and try again
            time.sleep(0.01)
            mplot_attempts += 1

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


def write_blade_file(name: str, base_dir: str, grid_bounds: typing.Iterable, coords: typing.List[np.ndarray],
                     mea_airfoil_names: typing.List[str]):
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

    coords: typing.List[np.ndarray]
      A 3-D set of coordinates to write as the airfoil geometry. The array of coordinates has size
      :math:`M \times N \times 2` where :math:`M` is the number of airfoils and :math:`N` is the number of airfoil
      coordinates. The coordinates can be input as a ragged array, where :math:`N` changes with each 3-D slice (i.e.,
      the number of airfoil coordinates can be different for each airfoil).

    Returns
    =======
    str
        Absolute path to the generated MSES blade file
    """
    # if not os.path.exists(os.path.join(base_dir, name)):  # if specified directory doesn't exist,
    #     os.mkdir(os.path.join(base_dir, name))  # create it
    # blade_file = os.path.join(base_dir, name, 'blade.' + name)  # blade file stored as
    # # <base_dir>\<name>\blade.<name>
    # with open(blade_file, 'w') as f:  # Open the blade_file with write permission
    #     f.write(name + '\n')  # Write the name of the airfoil on the first line
    #     for gb in grid_bounds[0:-1]:  # For each of the entries in grid_bounds except the last (type: list),
    #         f.write(str(gb) + " ")  # write the grid_bound at the specified index with a space after it in blade_file
    #     f.write(str(grid_bounds[-1]) + "\n")
    #     for idx, airfoil_coords in enumerate(coords):
    #         for xy in airfoil_coords:
    #             f.write(f"{str(xy[0])} {str(xy[1])}\n")
    #         if idx != len(coords) - 1:
    #             f.write('999.0 999.0\n')  # Write 999.0 999.0 with a carriage return to indicate to MSES that another
    #             # airfoil follows
    # return blade_file

    # Set the default grid bounds value
    if grid_bounds is None:
        grid_bounds = [-5.0, 5.0, -5.0, 5.0]

    # Write the header (line 1: airfoil name, line 2: grid bounds values separated by spaces)
    header = name + "\n" + " ".join([str(gb) for gb in grid_bounds])

    # Determine the correct ordering for the airfoils. MSES expects airfoils to be ordered from top to bottom
    max_y = [np.max(coord_xy[:, 1]) for coord_xy in coords]
    airfoil_order = np.argsort(max_y)[::-1]

    # Loop through the airfoils in the correct order
    mea_coords = None
    for airfoil_idx in airfoil_order:
        airfoil_coords = coords[airfoil_idx]  # Extract the airfoil coordinates for this airfoil
        if mea_coords is None:
            mea_coords = airfoil_coords
        else:
            mea_coords = np.row_stack((mea_coords, np.array([999.0, 999.0])))  # MSES-specific airfoil delimiter
            mea_coords = np.row_stack((mea_coords, airfoil_coords))  # Append this airfoil's coordinates to the mat.

    # Generate the full file path
    blade_file_path = os.path.join(base_dir, name, f"blade.{name}")

    # Save the coordinates to file
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            np.savetxt(blade_file_path, mea_coords, header=header, comments="")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

    # Get the airfoil name order
    airfoil_name_order = [mea_airfoil_names[idx] for idx in airfoil_order]

    return blade_file_path, airfoil_name_order


def write_gridpar_file(name: str, base_folder: str, mset_settings: dict, airfoil_name_order: typing.List[str]):
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
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
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

                for a in airfoil_name_order:
                    f.write(f"{multi_airfoil_grid[a]['dsLE_dsAvg']} {multi_airfoil_grid[a]['dsTE_dsAvg']} "
                            f"{multi_airfoil_grid[a]['curvature_exp']}\n")

                for a in airfoil_name_order:
                    f.write(f"{multi_airfoil_grid[a]['U_s_smax_min']} {multi_airfoil_grid[a]['U_s_smax_max']} "
                            f"{multi_airfoil_grid[a]['L_s_smax_min']} {multi_airfoil_grid[a]['L_s_smax_max']} "
                            f"{multi_airfoil_grid[a]['U_local_avg_spac_ratio']} {multi_airfoil_grid[a]['L_local_avg_spac_ratio']}\n")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

    return gridpar_file


def write_mses_file(name: str, base_folder: str, mses_settings: dict, airfoil_name_order: typing.List[str]):
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

    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
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

                for idx, airfoil_name in enumerate(airfoil_name_order):
                    f.write(f"{F['XTRSupper'][airfoil_name]} {F['XTRSlower'][airfoil_name]}")
                    if idx == len(airfoil_name_order) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

                f.write(f"{F['MCRIT']} {F['MUCON']}\n")

                if any([bool(flag) for flag in F['AD_flags']]) or bool(F['inverse_flag']):
                    f.write(f"{int(F['ISMOVE'])} {int(F['ISPRES'])}\n")
                    f.write(f"{int(F['NMODN'])} {int(F['NPOSN'])}\n")

                for idx, flag in enumerate(F['AD_flags']):
                    if bool(flag):
                        f.write(f"{int(F['ISDELH'][idx])} {F['XCDELH'][idx]} {F['PTRHIN'][idx]} {F['ETAH'][idx]}\n")
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1

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
    attempt = 0
    max_attempts = 100
    while attempt < max_attempts:
        try:
            with open(input_file, 'w') as f:
                for input_ in input_list:
                    f.write(input_)
                    f.write('\n')
            break
        except OSError:
            time.sleep(0.01)
            attempt += 1


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
        K_array = np.array(bl_data_dict["K"])
        bl_at_point = {
            "x": x_point,
            "K": np.interp(x_point, x_array, K_array),
        }
        return bl_at_point

    # Approximate the boundary layer thickness by taking the distance from the intersection point on the airfoil surface
    # to the intersection point on the boundary layer edge along the extraction line (the true value would be taking it
    # as the perpendicular distance from the surface. For small surface angles, this is a good approximation.)
    if hasattr(lower_bl_inters, "x") and hasattr(lower_bl_inters, "y"):
        bl_at_point_lower = evaluate_bl_at_point(bl_data_lower, lower_bl_inters.x)
    else:
        bl_at_point_lower = None

    if hasattr(upper_bl_inters, "x") and hasattr(upper_bl_inters, "y"):
        bl_at_point_upper = evaluate_bl_at_point(bl_data_upper, upper_bl_inters.x)
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


def line_integral_CPK_inviscid_old(Cp: np.ndarray, rho: np.ndarray, u: np.ndarray,
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


def line_integral_Edot_inviscid(Cp: np.ndarray, rho: np.ndarray, u: np.ndarray,
                                v: np.ndarray, V: np.ndarray, x: np.ndarray, y: np.ndarray, n_hat_dir: str):
    """
    ... according to
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
    if n_hat_dir not in ["up", "down", "left", "right"]:
        raise ValueError("Invalid direction for n_hat")

    perp_angle_dict = {
        "right": -np.pi / 2,
        "left": np.pi / 2,
        "up": -np.pi / 2,
        "down": np.pi / 2
    }

    # Calculate the direction of n_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        # if n_hat_dir in ["right", "left"]:
            # angle = np.arctan2(np.gradient(y, x), 1)
        # angle = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
        # else:
        angle = np.arctan2(1, np.gradient(x, y))

        # temp_angle = np.zeros(angle.shape)
        # for idx, a in enumerate(angle):
        #     if np.isnan(a):
        #         if idx == 0:
        #             temp_angle[idx] = angle[idx + 1]
        #         else:
        #             temp_angle[idx] = angle[idx - 1]
        #     else:
        #         temp_angle[idx] = angle[idx]
        # angle = temp_angle

    perp_angle = perp_angle_dict[n_hat_dir]
    n_hat = np.column_stack((np.cos(angle + perp_angle), np.sin(angle + perp_angle)))
    V_vec = np.column_stack((u, v))

    # Compute the dot product V_vec * n_hat
    dot_product = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec, n_hat)])
    # integrand = (-Cp - rho * V**2 + rho) * dot_product.flatten()
    integrand = (Cp + rho * (V**2 - 1)) * dot_product.flatten()

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


def line_integral_Edota_inviscid_TP(rho: np.ndarray, u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    ... according to
    Drela's "Power Balance in Aerodynamic Flows", a 2009 AIAA Journal article

    Parameters
    ==========
    rho: np.ndarray
        Node-centered density field

    u: np.ndarray
        Node-centered x-velocity field

    v: np.ndarray
        Node-centered y-velocity field

    x: np.ndarray
        1-d array of x-coordinates of the line along which to compute the line integral

    y: np.ndarray
        1-d array of y-coordinates of the line along which to compute the line integral
    """
    # Calculate the direction of n_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.arctan2(1, np.gradient(x, y))

    perp_angle = -np.pi / 2
    n_hat = np.column_stack((np.cos(angle + perp_angle), np.sin(angle + perp_angle)))
    V_vec = np.column_stack((u, v))

    # Compute the dot product V_vec * n_hat
    dot_product = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec, n_hat)])
    u_perturb = dot_product.flatten() - 1
    # print(f"{u_perturb = }")
    integrand = rho * u_perturb ** 2 * (1 + u_perturb)

    # Build the length increment vector (dl)
    dl = np.array([0.0])
    dl = np.append(dl, np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]))  # compute incremental length along x and y
    dl = np.cumsum(dl)

    # Integrate
    integral = np.trapz(integrand, dl)
    return integral


def line_integral_Edotp_inviscid_TP(Cp: np.ndarray, u: np.ndarray, v: np.ndarray, x: np.ndarray,
                                    y: np.ndarray):
    """
    ... according to
    Drela's "Power Balance in Aerodynamic Flows", a 2009 AIAA Journal article

    Parameters
    ==========
    Cp: np.ndarray
        Node-centered pressure coefficient field

    u: np.ndarray
        Node-centered x-velocity field

    v: np.ndarray
        Node-centered y-velocity field

    x: np.ndarray
        1-d array of x-coordinates of the line along which to compute the line integral

    y: np.ndarray
        1-d array of y-coordinates of the line along which to compute the line integral
    """
    # Calculate the direction of n_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.arctan2(1, np.gradient(x, y))

    perp_angle = -np.pi / 2
    n_hat = np.column_stack((np.cos(angle + perp_angle), np.sin(angle + perp_angle)))
    V_vec = np.column_stack((u, v))

    # Compute the dot product V_vec * n_hat
    dot_product = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec, n_hat)])
    u_perturb = dot_product.flatten() - 1
    integrand = Cp * u_perturb

    # Build the length increment vector (dl)
    dl = np.array([0.0])
    dl = np.append(dl, np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]))  # compute incremental length along x and y
    dl = np.cumsum(dl)

    # Integrate
    integral = np.trapz(integrand, dl)
    return integral


def viscous_Edot(last_BL_point: typing.Dict[str, np.ndarray]):
    """
    Eq. (75) of Drela's Power Balance in Aerodynamic Flows in 2-D form
    """
    if any([k not in last_BL_point.keys() for k in ("rhoe/rhoinf", "Ue/Uinf", "theta", "theta*")]):
        raise ValueError(f"Missing a boundary layer key in the Edot calculation. {last_BL_point.keys() = }")

    delta_K = 2 * last_BL_point["theta"] - last_BL_point["theta*"]
    Edot_visc = np.sum(last_BL_point["rhoe/rhoinf"] * last_BL_point["Ue/Uinf"]**3 * delta_K)
    return Edot_visc


def surface_dissipation(last_BL_point: typing.Dict[str, np.ndarray]):
    """
    Eq. (73) of Drela's Power Balance in Aerodynamic Flows
    """
    if any([k not in last_BL_point.keys() for k in ("K",)]):
        raise ValueError(f"Missing a boundary layer key in the Edot calculation. {last_BL_point.keys() = }")

    diss = np.sum(2 * last_BL_point["K"])
    return diss


def shock_dissipation(x_edge: typing.List[np.ndarray], y_edge: typing.List[np.ndarray], u: np.ndarray,
                      v: np.ndarray, M: np.ndarray, Cpt: np.ndarray, dCpt: np.ndarray, dCp: np.ndarray, gam: float):
    Shock = namedtuple("Shock", ("j", "i_start", "i_end"))
    shock_indices = np.argwhere(dCpt < -0.002)
    shock_indices = np.fliplr(shock_indices)
    shock_indices = shock_indices[shock_indices[:, 1].argsort()]
    shock_indices = shock_indices[shock_indices[:, 0].argsort(kind="mergesort")]
    shocks = {k: [] for k in np.unique(shock_indices[:, 0])}

    i_start = 0
    for idx, row in enumerate(shock_indices):
        j = row[0]
        if idx == 0:
            i_start = row[1]
            continue
        if row[0] == shock_indices[idx, 0] and row[1] == shock_indices[idx - 1, 1] + 1:
            pass
        else:
            i_end = shock_indices[idx - 1, 1] + 1
            shocks[j].append(Shock(j=j, i_start=i_start, i_end=i_end))
            i_start = row[1]

    i1 = 0
    edge_split_i = []
    for _idx, edge_array in enumerate(x_edge):
        i2 = i1 + edge_array.shape[1] - 2
        edge_split_i.append([i1, i2])
        i1 = i2 + 1

    # print(f"{edge_split_i = }")

    def integral(s: Shock):

        _dCp = dCp[s.i_start:s.i_end, s.j]

        if np.max(_dCp) < 0.1:  # If the static pressure rise is too small, do not consider this to be a shock wave
            return 0.0, 0.0

        # print(f"{_dCp = }, {s.i_start = }, {s.i_end = }")
        #
        # M_temp = M[s.i_start:s.i_end, s.j]
        # print(f"{M_temp = }")

        upstream_shock_cell_index = np.arange(s.i_start, s.i_end)[np.argwhere(_dCp > 0.0).flatten()[0]]
        # print(f"{upstream_shock_cell_index = }, {s.j = }")

        # max_Mach_number_search_iter = 7
        # iter_count = 0
        # old_M = M[upstream_shock_cell_index, s.j]
        # while True:
        #     print(f"{iter_count = }")
        #     if iter_count > max_Mach_number_search_iter:
        #         break
        #     upstream_shock_cell_index -= 1
        #     current_M = M[upstream_shock_cell_index, s.j]
        #     print(f"{old_M = }, {current_M = }")
        #     if current_M < old_M:
        #         upstream_shock_cell_index += 1
        #         break
        #
        #     old_M = current_M
        #     iter_count += 1

        _u = u[upstream_shock_cell_index, s.j]
        _v = v[upstream_shock_cell_index, s.j]
        _M = M[upstream_shock_cell_index, s.j]
        _Cpt = Cpt[upstream_shock_cell_index, s.j]

        j_flow = 0
        j_grid = 0
        for flow_idx, split_idx in enumerate(edge_split_i):
            if split_idx[0] <= s.j <= split_idx[1]:
                j_flow = flow_idx
                j_grid = s.j - split_idx[0]
                break

        x1 = x_edge[j_flow][upstream_shock_cell_index + 1, j_grid]
        x2 = x_edge[j_flow][upstream_shock_cell_index + 1, j_grid + 1]
        y1 = y_edge[j_flow][upstream_shock_cell_index + 1, j_grid]
        y2 = y_edge[j_flow][upstream_shock_cell_index + 1, j_grid + 1]
        n_hat_angle = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2
        n_hat = np.array([np.cos(n_hat_angle), np.sin(n_hat_angle)])

        V_dir = np.array([_u, _v]) / np.hypot(_u, _v)
        M_vec = _M * V_dir
        M_normal_up = np.dot(M_vec, n_hat)

        if isinstance(M_normal_up, np.ndarray):
            M_normal_up = M_normal_up[0]

        if M_normal_up <= 1.0:
            return 0.0, 0.0
        else:
            # print(f"{M_normal_up = }, {x1 = }, {y1 = }, {s.j = }, {upstream_shock_cell_index = }")
            Cpt2_Cpt1 = calculate_normal_shock_total_pressure_ratio(M_normal_up, gam)
            _dCpt = _Cpt * Cpt2_Cpt1 - _Cpt

        dot_product = np.dot(np.array([_u, _v]), n_hat)
        dl = np.hypot(x2 - x1, y2 - y1)
        shock_int = np.abs(_dCpt) * dot_product * dl
        return shock_int, dl

    # print(f"{shock_indices = }")

    shock_diss = 0.0
    dl_total = 0.0
    for shock_list in shocks.values():
        for shock in shock_list:
            s_int, incremental_dl = integral(shock)
            shock_diss += s_int
            dl_total += incremental_dl

    # print(f"{dl_total = }")

    return shock_diss


def line_integral_CPK_inviscid(Cp_up: np.ndarray, Cp_down: np.ndarray, rho_up: np.ndarray, rho_down: np.ndarray,
                               u_up: np.ndarray, u_down: np.ndarray,
                               v_up: np.ndarray, v_down: np.ndarray, V_up: np.ndarray, V_down: np.ndarray,
                               x: np.ndarray, y: np.ndarray):
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
    """
    # Calculate edge center locations
    x_edge_center_up = np.array([(x1 + x2) / 2 for x1, x2 in zip(x[0, :], x[1, :])])
    x_edge_center_down = np.array([(x1 + x2) / 2 for x1, x2 in zip(x[1, :], x[2, :])])
    y_edge_center_up = np.array([(y1 + y2) / 2 for y1, y2 in zip(y[0, :], y[1, :])])
    y_edge_center_down = np.array([(y1 + y2) / 2 for y1, y2 in zip(y[1, :], y[2, :])])

    # Calculate normals
    dx_dy_up = []
    dx_dy_down = []
    for i in range(len(x_edge_center_up) - 1):
        dx_dy_up.append((x_edge_center_up[i + 1] - x_edge_center_up[i]) / (
                y_edge_center_up[i + 1] - y_edge_center_up[i]))
    for i in range(len(x_edge_center_down) - 1):
        dx_dy_down.append((x_edge_center_down[i + 1] - x_edge_center_down[i]) / (
                y_edge_center_down[i + 1] - y_edge_center_down[i]))
    dx_dy_up = np.array(dx_dy_up)
    dx_dy_down = np.array(dx_dy_down)

    angle_up = np.arctan2(1, dx_dy_up)
    angle_down = np.arctan2(1, dx_dy_down)

    perp_angle_up = -np.pi / 2
    perp_angle_down = np.pi / 2

    n_hat_up = np.column_stack((np.cos(angle_up + perp_angle_up), np.sin(angle_up + perp_angle_up)))
    n_hat_down = np.column_stack((np.cos(angle_down + perp_angle_down), np.sin(angle_down + perp_angle_down)))

    # Generate velocity vectors
    V_vec_up = np.column_stack((u_up, v_up))
    V_vec_down = np.column_stack((u_down, v_down))

    # Calculate dL_B/c_main
    dl_up = []
    dl_down = []

    for i in range(len(x_edge_center_up) - 1):
        dl_up.append(np.hypot(x_edge_center_up[i + 1] - x_edge_center_up[i], y_edge_center_up[i + 1] - y_edge_center_up[i]))
    for i in range(len(x_edge_center_down) - 1):
        dl_down.append(np.hypot(x_edge_center_down[i + 1] - x_edge_center_down[i], y_edge_center_down[i + 1] - y_edge_center_down[i]))

    dl_up = np.array(dl_up)
    dl_down = np.array(dl_down)

    dl_up = np.cumsum(dl_up)
    dl_down = np.cumsum(dl_down)

    # Compute the dot product V_vec * n_hat
    dot_product_up = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec_up, n_hat_up)])
    dot_product_down = np.array([np.dot(V_vec_i, n_hat_i) for V_vec_i, n_hat_i in zip(V_vec_down, n_hat_down)])

    # Compute the integrand
    integrand_up = (rho_up * (1 - V_up**2) - Cp_up) * dot_product_up.flatten()
    integrand_down = (rho_down * (1 - V_down**2) - Cp_down) * dot_product_down.flatten()

    # Integrate
    integral_up = np.trapz(integrand_up, dl_up)
    integral_down = np.trapz(integrand_down, dl_down)
    integral = integral_up + integral_down

    return integral


def line_integral_CPK_bl(rhoe_rhoinf, ue_uinf, Cp, deltastar, outlet: bool):
    r"""
    Computes CPK at a given boundary layer location

    Parameters
    ==========
    K: float
        Non-dimensional kinetic energy thickness, :math:`0.5 \frac{\rho_e}{\rho_\infty} \left( \frac{u_e}{V_\infty} \right)^3 \theta^*`

    outlet: bool
        Whether this boundary layer is part of an outlet
    """
    # 7/8 * edge velocity is an approximate average velocity based on a 1/7 power law velocity profile
    dot_product = -7/8 * ue_uinf if outlet else 7/8 * ue_uinf
    integrand = (rhoe_rhoinf * (1 - (ue_uinf * 7/8) ** 2) - Cp) * dot_product
    dl = deltastar
    integral = integrand * dl
    return integral


def calculate_CPK_mses_old(analysis_subdir: str, configuration: str = "underwing_te", calculate_capSS: bool = False,
                           calculate_exit_plane_Mach_array: bool = False):
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
    mses_log_file = os.path.join(analysis_subdir, "mses.log")
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

    hub_te_upper = coords[1][0, :]
    hub_te_lower = coords[1][-1, :]

    angle = np.arctan2(main_te_lower[1] - nacelle_te_upper[1], main_te_lower[0] - nacelle_te_upper[0])

    plot_planes = False
    # PLOT INLET AND OUTLET PLANES
    if plot_planes:
        for c_set in coords:
            plt.plot(c_set[:, 0], c_set[:, 1])
        plt.plot(np.array([chord_5perc[0], chord_5perc[0] + 0.3 * np.cos(angle)]),
                 np.array([chord_5perc[1], chord_5perc[1] + 0.3 * np.sin(angle)]))
        plt.plot(np.array([hub_te_upper[0], hub_te_upper[0] + 0.3 * np.cos(angle)]),
                 np.array([hub_te_upper[1], hub_te_upper[1] + 0.3 * np.sin(angle)]))
        plt.plot(np.array([hub_te_lower[0], hub_te_lower[0] - 0.3 * np.cos(angle)]),
                 np.array([hub_te_lower[1], hub_te_lower[1] - 0.3 * np.sin(angle)]))
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

    capSS = None
    epma = []  # Exit plane Mach array

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
                                                       field[flow_var_idx["V"]][:, start_idx:end_idx])
            M = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                       field[flow_var_idx["M"]][:, start_idx:end_idx])

            if calculate_capSS or calculate_exit_plane_Mach_array:
                if capSS in [None, -1]:
                    capSS = -1
                    if np.any(M > 1.0):
                        capSS = np.sum((M[M > 1.0]) ** 2) ** 0.5

            bl_data_lower = bl_data[underwing_flow_section_bl_map[flow_section_idx][0]]
            bl_data_upper = bl_data[underwing_flow_section_bl_map[flow_section_idx][1]]

            inlet_point = chord_5perc
            outlet_point = nacelle_te_upper
            # outlet_point = hub_te_upper

            xyCprhouvVM_out, bl_at_point_upper_out, bl_at_point_lower_out = extrapolate_data_line_mses_field(
                [Cp, rho, u, v, V, M], x_grid[flow_section_idx], y_grid[flow_section_idx],
                bl_data_lower=bl_data_lower, bl_data_upper=bl_data_upper, point=outlet_point, angle=angle
            )

            xyCprhouvV_in, bl_at_point_upper_in, bl_at_point_lower_in = extrapolate_data_line_mses_field(
                [Cp, rho, u, v, V], x_grid[flow_section_idx], y_grid[flow_section_idx],
                bl_data_lower=bl_data_lower, bl_data_upper=bl_data_upper, point=inlet_point, angle=angle
            )

            # Integrate over the propulsor outlet for the given flow section
            Cp_out, rho_out, u_out, v_out, V_out, M_out = \
                xyCprhouvVM_out[:, 2], xyCprhouvVM_out[:, 3], xyCprhouvVM_out[:, 4], xyCprhouvVM_out[:, 5], \
                xyCprhouvVM_out[:, 6], xyCprhouvVM_out[:, 7]
            x_out = xyCprhouvVM_out[:, 0]
            y_out = xyCprhouvVM_out[:, 1]
            outlet_integral = line_integral_CPK_inviscid_old(Cp_out, rho_out, u_out, v_out, V_out, x_out, y_out,
                                                         n_hat_right=False)  # n_hat points into the propulsor
            CPK += outlet_integral
            # print(f"{outlet_integral = }")
            epma.extend(M_out.tolist())

            if bl_at_point_upper_out is not None:
                # print(f"{bl_at_point_upper_out['K'] = }")
                CPK += line_integral_CPK_bl(bl_at_point_upper_out["K"], outlet=True)

            if bl_at_point_lower_out is not None:
                # print(f"{bl_at_point_lower_out['K'] = }")
                CPK += line_integral_CPK_bl(bl_at_point_lower_out["K"], outlet=True)

            # print(f"After adding the BL CPK, {CPK - outlet_integral = }")

            # Integrate over the propulsor inlet for the given flow section
            Cp_in, rho_in, u_in, v_in, V_in = \
                xyCprhouvV_in[:, 2], xyCprhouvV_in[:, 3], xyCprhouvV_in[:, 4], xyCprhouvV_in[:, 5], xyCprhouvV_in[:, 6]
            x_in = xyCprhouvV_in[:, 0]
            y_in = xyCprhouvV_in[:, 1]
            inlet_integral = line_integral_CPK_inviscid_old(Cp_in, rho_in, u_in, v_in, V_in, x_in, y_in,
                                                        n_hat_right=True)  # n_hat points into the propulsor
            CPK += inlet_integral
            # CPK_temp = CPK
            # print(f"{inlet_integral = }, {CPK = }")

            if bl_at_point_upper_in is not None:
                # print(f"{bl_at_point_upper_in['K'] = }")
                CPK += line_integral_CPK_bl(bl_at_point_upper_in["K"], outlet=False)

            if bl_at_point_lower_in is not None:
                # print(f"{bl_at_point_lower_in['K'] = }")
                CPK += line_integral_CPK_bl(bl_at_point_lower_in["K"], outlet=False)

            # print(f"After adding the inlet BL contributions, {CPK - CPK_temp = }")

        if flow_section_idx < grid_stats["numel"]:
            start_idx = end_idx
            end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

    if np.isnan(CPK):
        CPK = 1e9

    # print(f"{CPK = }, {inlet_integral = }, {outlet_integral = }")

    # if calculate_capSS:
    #     return CPK, capSS
    # else:
    #     return CPK

    return {"CPK": CPK, "capSS": capSS, "epma": epma}


def calculate_CPK_power_consumption_old(analysis_subdir: str):
    """
    A specialized function that calculates the mechanical flow power coefficient for an underwing trailing edge
    aero-propulsive configuration.
    """
    airfoil_system_name = os.path.split(analysis_subdir)[-1]
    field_file = os.path.join(analysis_subdir, f'field.{airfoil_system_name}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{airfoil_system_name}')
    blade_file = os.path.join(analysis_subdir, f"blade.{airfoil_system_name}")
    bl_file = os.path.join(analysis_subdir, f"bl.{airfoil_system_name}")
    mses_file = os.path.join(analysis_subdir, f"mses.{airfoil_system_name}")
    mses_log_file = os.path.join(analysis_subdir, "mses.log")
    coords = convert_blade_file_to_3d_array(blade_file)
    mplot_log_file = os.path.join(analysis_subdir, "mplot.log")

    M_inf = read_Mach_from_mses_file(mses_file)
    gam = 1.4  # Hard-coded specific heat ratio
    forces = read_forces_from_mses(mplot_log_file)
    field = read_field_from_mses(field_file, M_inf=M_inf, gam=gam)
    bl_data = read_bl_data_from_mses(bl_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)

    Edot = 0.0

    start_idx, end_idx = 0, x_grid[0].shape[1] - 1

    for flow_section_idx in range(grid_stats["numel"] + 1):
        Cp = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["Cp"]][:, start_idx:end_idx])
        rho = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                     field[flow_var_idx["rho"]][:, start_idx:end_idx])
        u = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["u"]][:, start_idx:end_idx])
        v = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["v"]][:, start_idx:end_idx])
        V = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["V"]][:, start_idx:end_idx])
        x = x_grid[flow_section_idx]
        y = y_grid[flow_section_idx]

        # Integrate over the propulsor outlet for the given flow section

        outlet_integral = line_integral_Edot_inviscid(Cp[-1, :], rho[-1, :], u[-1, :], v[-1, :], V[-1, :], x[-1, :],
                                             y[-1, :], n_hat_dir="right")
        Edot += outlet_integral

        # print(f"{outlet_integral = }")

        # The side cylinder contributions are negligible
        # if flow_section_idx == len(x_grid) - 1:
        #
        #     top_integral = line_integral_Edot_inviscid(Cp[:, -1], rho[:, -1], u[:, -1], v[:, -1], V[:, -1], x[:, -1],
        #                                         y[:, -1], n_hat_dir="up")
        #     CPK += top_integral
        #     print(f"{top_integral = }")
        #
        # elif flow_section_idx == 0:
        #     bottom_integral = line_integral_Edot_inviscid(Cp[:, 0], rho[:, 0], u[:, 0], v[:, 0], V[:, 0], x[:, 0],
        #                                          y[:, 0], n_hat_dir="down")
        #     CPK += bottom_integral
        #     print(f"{bottom_integral = }")

        # print(f"{outlet_integral = }, {inlet_integral = }, {Edot = }")

        # print(f"After adding the inlet BL contributions, {CPK - CPK_temp = }")

        if flow_section_idx < grid_stats["numel"]:
            start_idx = end_idx
            end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

    end_point_counter = 0
    last_point_idx = -1
    max_end_point_attempts = 10

    while True:
        if end_point_counter > max_end_point_attempts:
            raise ValueError("Reached maximum attempts for setting the boundary layer end point.")

        last_BL_point = {}
        for side in bl_data:
            for k in ("rhoe/rhoinf", "Ue/Uinf", "theta", "theta*", "K"):
                if k not in last_BL_point.keys():
                    last_BL_point[k] = []
                val = side[k][last_point_idx]
                if val <= 1e-12 and k == "K":
                    end_point_counter += 1
                    # print(f"Decrementing last_point_idx...")
                    last_point_idx -= 1
                    continue
                last_BL_point[k].append(val)

        for k, v in last_BL_point.items():
            last_BL_point[k] = np.array(v)
        break

    # print(f"{Edot = }")

    Edot_visc = viscous_Edot(last_BL_point=last_BL_point)

    Edot += Edot_visc

    # print(f"{Edot_visc = }")

    surface_diss = surface_dissipation(last_BL_point=last_BL_point)

    # print(f"{surface_diss = }")

    shock_diss = shock_dissipation(x_edge=x_grid, y_edge=y_grid, u=field[flow_var_idx["u"]][:, :],
                                   v=field[flow_var_idx["v"]][:, :], M=field[flow_var_idx["M"]][:, :],
                                   Cpt=field[flow_var_idx["Cpt"]][:, :], dCpt=field[flow_var_idx["dCpt"]][:, :],
                                   dCp=field[flow_var_idx["dCp"]][:, :], gam=gam)

    # print(f"{shock_diss = }")
    # print(f"{forces['Cdw'] = }, {forces['Cd'] = }")

    CPK = Edot + surface_diss + shock_diss - forces["Cd"]

    # print(f"{CPK = }")

    if np.isnan(CPK):
        CPK = 1e9

    # print(f"{CPK = }, {inlet_integral = }, {outlet_integral = }")

    # if calculate_capSS:
    #     return CPK, capSS
    # else:
    #     return CPK

    return {"CPK": CPK, "Edot": Edot, "diss_surf": surface_diss, "diss_shock": shock_diss, "Cd": forces["Cd"]}


def calculate_CPK_power_consumption(analysis_subdir: str):
    """
    A specialized function that calculates the mechanical flow power coefficient
    """
    airfoil_system_name = os.path.split(analysis_subdir)[-1]
    field_file = os.path.join(analysis_subdir, f'field.{airfoil_system_name}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{airfoil_system_name}')
    blade_file = os.path.join(analysis_subdir, f"blade.{airfoil_system_name}")
    bl_file = os.path.join(analysis_subdir, f"bl.{airfoil_system_name}")
    mses_file = os.path.join(analysis_subdir, f"mses.{airfoil_system_name}")
    mses_log_file = os.path.join(analysis_subdir, "mses.log")
    coords = convert_blade_file_to_3d_array(blade_file)
    mplot_log_file = os.path.join(analysis_subdir, "mplot.log")

    M_inf = read_Mach_from_mses_file(mses_file)
    gam = 1.4  # Hard-coded specific heat ratio
    forces = read_forces_from_mses(mplot_log_file)
    field = read_field_from_mses(field_file, M_inf=M_inf, gam=gam)
    bl_data = read_bl_data_from_mses(bl_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)

    Edot = 0.0
    Edota = 0.0
    Edotp = 0.0

    start_idx, end_idx = 0, x_grid[0].shape[1] - 1

    for flow_section_idx in range(grid_stats["numel"] + 1):
        Cp = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["Cp"]][:, start_idx:end_idx])
        rho = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                     field[flow_var_idx["rho"]][:, start_idx:end_idx])
        u = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["u"]][:, start_idx:end_idx])
        v = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["v"]][:, start_idx:end_idx])
        V = convert_cell_centered_to_edge_centered(x_grid[flow_section_idx].shape,
                                                   field[flow_var_idx["V"]][:, start_idx:end_idx])
        x = x_grid[flow_section_idx]
        y = y_grid[flow_section_idx]

        # Integrate over the propulsor outlet for the given flow section

        # outlet_integral = line_integral_Edot_inviscid(Cp[-1, :], rho[-1, :], u[-1, :], v[-1, :], V[-1, :], x[-1, :],
        #                                      y[-1, :], n_hat_dir="right")
        #
        # Edot += outlet_integral

        Edota += line_integral_Edota_inviscid_TP(rho[-1, :], u[-1, :], v[-1, :], x[-1, :], y[-1, :])
        Edotp += line_integral_Edotp_inviscid_TP(Cp[-1, :], u[-1, :], v[-1, :], x[-1, :], y[-1, :])
        # print(f"{Edota = }, {Edotp = }, {np.mean(rho[-1, :]) = }, {np.max(rho[-1, :]) = }, {np.min(rho[-1, :]) = }")
        # print(f"{np.mean(Cp[-1, :]) = }, {np.max(Cp[-1, :]) = }, {np.min(Cp[-1, :]) = }")
        # print(f"{np.mean(u[-1, :]) = }, {np.max(u[-1, :]) = }, {np.min(u[-1, :]) = }")
        # print(f"{np.mean(v[-1, :]) = }, {np.max(v[-1, :]) = }, {np.min(v[-1, :]) = }")
        # print(f"{np.mean(x[-1, :]) = }, {np.max(x[-1, :]) = }, {np.min(x[-1, :]) = }")
        # print(f"{np.mean(y[-1, :]) = }, {np.max(y[-1, :]) = }, {np.min(y[-1, :]) = }")

        if flow_section_idx < grid_stats["numel"]:
            start_idx = end_idx
            end_idx += x_grid[flow_section_idx + 1].shape[1] - 1

    end_point_counter = 0
    last_point_idx = -1
    max_end_point_attempts = 10

    while True:
        if end_point_counter > max_end_point_attempts:
            raise ValueError("Reached maximum attempts for setting the boundary layer end point.")

        last_BL_point = {}
        for side in bl_data:
            for k in ("rhoe/rhoinf", "Ue/Uinf", "theta", "theta*", "K"):
                if k not in last_BL_point.keys():
                    last_BL_point[k] = []
                val = side[k][last_point_idx]
                if val <= 1e-12 and k == "K":
                    end_point_counter += 1
                    # print(f"Decrementing last_point_idx...")
                    last_point_idx -= 1
                    continue
                last_BL_point[k].append(val)

        for k, v in last_BL_point.items():
            last_BL_point[k] = np.array(v)
        break

    Edota_visc = viscous_Edot(last_BL_point=last_BL_point)

    Edota += Edota_visc

    # TODO: need to find a way to add the pressure component from the boundary layer region (Edotp_visc)

    surface_diss = surface_dissipation(last_BL_point=last_BL_point)

    shock_diss = shock_dissipation(x_edge=x_grid, y_edge=y_grid, u=field[flow_var_idx["u"]][:, :],
                                   v=field[flow_var_idx["v"]][:, :], M=field[flow_var_idx["M"]][:, :],
                                   Cpt=field[flow_var_idx["Cpt"]][:, :], dCpt=field[flow_var_idx["dCpt"]][:, :],
                                   dCp=field[flow_var_idx["dCp"]][:, :], gam=gam)

    # CPK = Edot + surface_diss + shock_diss - forces["Cd"]
    CPK = Edota + Edotp + surface_diss + shock_diss - forces["Cd"]

    if np.isnan(CPK):
        CPK = 1e9

    return {"CPK": CPK, "Edota": Edota, "Edotp": Edotp, "diss_surf": surface_diss, "diss_shock": shock_diss,
            "Cd": forces["Cd"]}


def calculate_CPK_mses_inviscid_only(analysis_subdir: str):
    """
    Calculates the mechanical flower power coefficient input to the control volume across the airfoil system control
    surface. Assumes that the control surface wraps just around the actuator disk and that the normal vectors point
    into the propulsor. Also assumes that there is no change in the kinetic energy defect across the actuator disk
    (and thus no change in CPK due to the boundary layer).
    """

    airfoil_system_name = os.path.split(analysis_subdir)[-1]
    field_file = os.path.join(analysis_subdir, f'field.{airfoil_system_name}')
    grid_stats_file = os.path.join(analysis_subdir, 'mplot_grid_stats.log')
    grid_file = os.path.join(analysis_subdir, f'grid.{airfoil_system_name}')
    mses_log_file = os.path.join(analysis_subdir, "mses.log")

    field = read_field_from_mses(field_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
    data_AD = read_actuator_disk_data_mses(mses_log_file, grid_stats)

    CPK = 0.0

    for data in data_AD:
        Cp_up = field[flow_var_idx["Cp"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        Cp_down = field[flow_var_idx["Cp"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_up = field[flow_var_idx["rho"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_down = field[flow_var_idx["rho"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        u_up = field[flow_var_idx["u"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        u_down = field[flow_var_idx["u"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        v_up = field[flow_var_idx["v"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        v_down = field[flow_var_idx["v"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        V_up = field[flow_var_idx["V"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        V_down = field[flow_var_idx["V"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        x = x_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        y = y_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        CPK += line_integral_CPK_inviscid(Cp_up, Cp_down, rho_up, rho_down, u_up, u_down,
                                          v_up, v_down, V_up, V_down, x, y)

    if np.isnan(CPK):
        CPK = 1e9

    return {"CPK": CPK}


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
    mses_log_file = os.path.join(analysis_subdir, "mses.log")
    bl_file = os.path.join(analysis_subdir, f"bl.{airfoil_system_name}")

    field = read_field_from_mses(field_file)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
    data_AD = read_actuator_disk_data_mses(mses_log_file, grid_stats)
    bl_data = read_bl_data_from_mses(bl_file)

    CPK = 0.0

    for data in data_AD:
        Cp_up = field[flow_var_idx["Cp"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        Cp_down = field[flow_var_idx["Cp"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_up = field[flow_var_idx["rho"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        rho_down = field[flow_var_idx["rho"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        u_up = field[flow_var_idx["u"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        u_down = field[flow_var_idx["u"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        v_up = field[flow_var_idx["v"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        v_down = field[flow_var_idx["v"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        V_up = field[flow_var_idx["V"]][data["field_i_up"], data["field_j_start"]:data["field_j_end"] + 1]
        V_down = field[flow_var_idx["V"]][data["field_i_down"], data["field_j_start"]:data["field_j_end"] + 1]
        x = x_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        y = y_grid[data["flow_section_idx"]][data["field_i_up"]:data["field_i_up"] + 3, :]
        CPK += line_integral_CPK_inviscid(Cp_up, Cp_down, rho_up, rho_down, u_up, u_down,
                                          v_up, v_down, V_up, V_down, x, y)

        # Calculate the bounding boundary layer sides corresponding to the given flow section index
        n_airfoils = len(x_grid) - 1
        side_lower = (n_airfoils - data["flow_section_idx"]) * 2
        side_upper = side_lower - 1

        # Retrieve the boundary layer data for each boundary layer side
        bl_data_dict = {
            "upstream_lower": bl_data[side_lower],
            "downstream_lower": bl_data[side_lower],
            "upstream_upper": bl_data[side_upper],
            "downstream_upper": bl_data[side_upper]
        }

        # Calculate edge center locations
        xe = {
            "upstream_lower": (x[0, 0] + x[1, 0]) / 2,
            "downstream_lower": (x[1, 0] + x[2, 0]) / 2,
            "upstream_upper": (x[0, -1] + x[1, -1]) / 2,
            "downstream_upper": (x[1, -1] + x[2, -1]) / 2
        }

        outlet_bool = {
            "upstream_lower": False,
            "downstream_lower": True,
            "upstream_upper": False,
            "downstream_upper": True,
        }

        # Interpolate the boundary layer data for each edge center location and calculate the CPK value
        for xe_k, xe_v in xe.items():
            bl = bl_data_dict[xe_k]
            rhoe_rhoinf = np.interp(xe_v, bl["x"], bl["rhoe/rhoinf"])
            ue_uinf = np.interp(xe_v, bl["x"], bl["Ue/Uinf"])
            Cp = np.interp(xe_v, bl["x"], bl["Cp"])
            deltastar = np.interp(xe_v, bl["x"], bl["delta*"])
            CPK_bl = line_integral_CPK_bl(rhoe_rhoinf=rhoe_rhoinf, ue_uinf=ue_uinf, Cp=Cp, deltastar=deltastar,
                                          outlet=outlet_bool[xe_k])
            CPK += CPK_bl

    if np.isnan(CPK):
        CPK = 1e9

    return {"CPK": CPK}


class GeometryError(Exception):
    pass
