import subprocess
import os
from pymead.analysis.read_aero_data import read_Cl_from_file_panel_fort, read_Cp_from_file_panel_fort, \
    read_aero_data_from_xfoil, read_Cp_from_file_xfoil, read_bl_data_from_mses, read_forces_from_mses
from pymead.utils.file_conversion import convert_ps_to_svg
from pymead.utils.geometry import check_airfoil_self_intersection
from pymead.utils.read_write_files import write_tuple_tuple_to_file
from pymead.optimization.opt_setup import update_mses_settings_from_stencil
import typing


SVG_PLOTS = ['Mach_contours', 'grid', 'grid_zoom']
SVG_SETTINGS_TR = {
    SVG_PLOTS[0]: 'Mach',
    SVG_PLOTS[1]: 'Grid',
    SVG_PLOTS[2]: 'Grid_Zoom',
}


def calculate_aero_data(airfoil_coord_dir: str, airfoil_name: str, coords: typing.Tuple[tuple],
                        tool: str = 'panel_fort', xfoil_settings: dict = None, mset_settings: dict = None,
                        mses_settings: dict = None, mplot_settings: dict = None, export_Cp: bool = True,
                        alpha_panel_fort=None):
    # ratio_thresh of 1.000005 and abs_thresh = 0.1 works well
    tool_list = ['panel_fort', 'XFOIL', 'MSES']
    if tool not in tool_list:
        raise ValueError(f"\'tool\' must be one of {tool_list}")
    if tool in ['panel_fort', 'XFOIL']:

        # Check for self-intersection and early return if self-intersecting:
        if check_airfoil_self_intersection(coords):
            return False

    aero_data = {}
    base_dir = os.path.join(airfoil_coord_dir, airfoil_name)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if tool == 'panel_fort':
        f = os.path.join(base_dir, airfoil_name + '.dat')
        n_data_pts = len(coords)
        write_tuple_tuple_to_file(f, coords)
        subprocess.run((["panel_fort", airfoil_coord_dir, airfoil_name + '.dat', str(n_data_pts - 1),
                         str(alpha_panel_fort)]),
                       stdout=subprocess.DEVNULL)  # stdout=subprocess.DEVNULL suppresses output to console
        aero_data['Cl'] = read_Cl_from_file_panel_fort(os.path.join(airfoil_coord_dir, 'LIFT.dat'))
        if export_Cp:
            aero_data['Cp'] = read_Cp_from_file_panel_fort(os.path.join(airfoil_coord_dir, 'CPLV.DAT'))
        return aero_data
    elif tool == 'XFOIL':
        if xfoil_settings is None:
            raise ValueError(f"\'xfoil_settings\' must be set if \'xfoil\' tool is selected")
        if 'xtr' not in xfoil_settings.keys():
            xfoil_settings['xtr'] = [1.0, 1.0]
        if 'N' not in xfoil_settings.keys():
            xfoil_settings['N'] = 9.0
        f = os.path.join(base_dir, airfoil_name + ".dat")
        write_tuple_tuple_to_file(f, coords)
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
        if 'multi_point_active' in mses_settings.keys() and mses_settings['multi_point_active']:
            stencil = mses_settings['multi_point_stencil']
            mset_mplot_loop_iterations = len(next(iter(stencil))['points'])
            aero_data_list = []

        # Multipoint Loop
        for i in range(mset_mplot_loop_iterations):

            if stencil is not None:
                mses_settings = update_mses_settings_from_stencil(mses_settings=mses_settings, stencil=stencil, idx=i)

            print(f"Now, {mses_settings = }")

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

                for mplot_output_name in ['Mach', 'Grid', 'Grid_Zoom', 'flow_field']:
                    if mplot_settings[mplot_output_name]:
                        run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode=mplot_output_name)
                        if mplot_output_name == 'flow_field':
                            run_mplot(airfoil_name, airfoil_coord_dir, mplot_settings, mode='grid_stats')

                if converged:
                    aero_data['converged'] = True
                    aero_data['timed_out'] = False
                    aero_data['errored_out'] = False

                if aero_data_list is not None:
                    aero_data_list.append(aero_data)

        logs = {'mset': mset_log, 'mses': mses_log, 'mplot': mplot_log}
        if aero_data_list is None:
            return aero_data, logs
        else:
            return aero_data_list, logs


def run_mset(name: str, base_dir: str, mset_settings: dict, coords: typing.Tuple[tuple]):
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


def write_blade_file(name: str, base_dir: str, grid_bounds, coords: typing.Tuple[tuple]):
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
    :param name: Name of the airfoil [system]
    :param base_folder: Location where the grid parameter file should be stored
    :param mset_settings: Parameter set (dictionary)
    :return: Path of the created grid parameter file (str)
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
    F = mses_settings
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


def write_input_file(input_file: str, input_list: list):
    with open(input_file, 'w') as f:
        for input_ in input_list:
            f.write(input_)
            f.write('\n')


def convert_xfoil_string_to_aero_data(line1: str, line2: str, aero_data: dict):
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


class GeometryError(Exception):
    pass
