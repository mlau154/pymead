import os
import re
import typing

import pandas as pd
import numpy as np


flow_var_idx = {'M': 7, 'Cp': 8, 'p': 3, 'rho': 2, 'u': 4, 'v': 5, 'q': 6}


def read_Cp_from_file_xfoil(fname: str):
    """
    Reads the :math:`C_p` data from the format output by XFOIL and converts it to a *numpy* array

    Parameters
    ==========
    fname: str
      File from which to read the :math:`C_p` data

    Returns
    =======
    dict
      Dictionary containing 1-D *numpy* arrays for :math:`x`, :math:`y`, and :math:`C_p`
    """
    with open(fname, "r") as f:
        first_line = f.readline()

    # Read in the Cp data for XFOIL versions 6.93 or 6.99 (other versions are untested)
    if "y" in first_line:  # For XFOIL 6.99
        df = pd.read_csv(fname, skiprows=3, names=['x', 'y', 'Cp'], sep='\s+', engine='python')
        array_ = df.to_numpy()
        return {'x': array_[:, 0], 'y': array_[:, 1], 'Cp': array_[:, 2]}
    else:  # For XFOIL 6.93
        df = pd.read_csv(fname, skiprows=1, names=['x', 'Cp'], sep='\s+', engine='python')
        array_ = df.to_numpy()
        return {'x': array_[:, 0], 'Cp': array_[:, 1]}


def read_aero_data_from_xfoil(fpath: str, aero_data: dict):
    """
    Reads aerodynamic data from XFOIL's output log

    Parameters
    ==========
    fpath: str
      File path to the XFOIL log

    aero_data: dict
      Dictionary in which the aerodynamic data will be written

    Returns
    =======
    str, str
      The first and second lines of the XFOIL log files containing aerodynamic output data
    """
    line1 = None
    line2 = None
    for line in read_reverse_order(fpath):
        if 'VISCAL:  Convergence failed' in line:
            aero_data['converged'] = False
            break
        elif 'Cm' in line:
            line1 = line
        elif 'CL' in line:
            line2 = line
            aero_data['converged'] = True
            break
    else:
        aero_data['converged'] = False
        aero_data['errored_out'] = True

    return line1, line2


def read_reverse_order(file_name):
    """
    Reads a file in reverse order. From `<https://thispointer.com/python-read-a-file-in-reverse-order-line-by-line/>`_
    """
    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Fetch the line from buffer and yield it
                yield buffer.decode()[::-1]
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            yield buffer.decode()[::-1]


def read_forces_from_mses(search_file: str):
    """
    This function uses regex to extract the aerodynamic force coefficients of the airfoil [system] from the
    ``list forces`` option available in the MPLOT menu

    Parameters
    ==========
    search_file: str
      "Forces" file to be read (log of the MPLOT function with ``mode=="forces"``)

    Returns
    =======
    dict
      Dictionary containing the aerodynamic force coefficients
    """
    CL_lines = []
    alpha_line, vw_line, fp_line, ad_line = None, None, None, None
    with open(search_file, 'r') as f:
        file_out = f.readlines()
    for line in file_out:
        if re.search("CL", line):
            CL_lines.append(line)
        if 'alpha' in line:
            alpha_line = line
        if 'viscous' in line:
            vw_line = line
        if 'friction' in line:
            fp_line = line
        if 'CDh' in line:
            ad_line = line
    try:
        required_line = CL_lines[1]  # line we need happens the second time the string "CL" is mentioned
        split_line = required_line.split()  # split the up the line to grab the actual numbers
        forces = {  # write as a dictionary
            'Cl': float(split_line[2]),  # CL = VALUE
            'Cd': float(split_line[5]),  # CD = VALUE
            'Cm': float(split_line[8])  # CM = VALUE
        }

        # Find the angle of attack
        alpha_line = alpha_line.split('=')
        alpha_line = alpha_line[-1].split()
        forces['alf'] = float(alpha_line[-2])

        # Find the viscous drag and wave drag coefficients in the drag breakdown (these two values + Cdh should add to Cd)
        vw_line = vw_line.split('=')
        forces['Cdv'] = float(vw_line[-2].split()[0])
        forces['Cdw'] = float(vw_line[-1].split()[0])

        # Find the friction drag and pressure drag coefficients in the drag breakdown (these two values should add to Cd)
        fp_line = fp_line.split('=')
        forces['Cdf'] = float(fp_line[-2].split()[0])
        forces['Cdp'] = float(fp_line[-1].split()[0])

        # Find the "drag" due to the actuator disk:
        if ad_line is not None:
            ad_line = ad_line.split('=')
            forces['Cdh'] = float(ad_line[-1].split()[0])
        else:
            forces['Cdh'] = 0.0
    except:
        forces = {'Cl': 0.0, 'Cd': 1000.0, 'Cm': 1000.0, 'alf': 0.0, 'Cdv': 1000.0, 'Cdw': 1000.0,
                  'Cdf': 1000.0, 'Cdp': 1000.0, 'Cdh': 1000.0, 'CPK': 1000.0}

    # print(f"{forces = }")
    return forces


def read_grid_stats_from_mses(src_file: str):
    """
    Reads grid statistics from an MPLOT output file and outputs them to a dictionary

    Parameters
    ==========
    src_file: str
      The file from which to read the grid statistics

    Returns
    =======
    dict
      The grid statistics in a Python dictionary format
    """
    with open(src_file, 'r') as f:
        lines = f.readlines()

    grid_stats = {'grid_size': None, 'numel': None, 'Jside1': [], 'Jside2': [],
                  'ILE1': [], 'ILE2': [], 'ITE1': [], 'ITE2': [], }

    for line in lines:
        if 'Grid size' in line:
            line_split = line.split('=')[-1]
            # print(f"{[idx for idx in line_split.split()] = }")
            grid_stats['grid_size'] = [int(idx) for idx in line_split.split()]
        elif 'Number' in line:
            line_split = line.split(':')[-1]
            grid_stats['numel'] = int(line_split.strip())
        elif 'Jside1' in line:
            line_split = line.split('=')
            grid_stats['Jside1'].append(int(line_split[1].split()[0]))
            grid_stats['ILE1'].append(int(line_split[2].split()[0]))
            grid_stats['ITE1'].append(int(line_split[3].split()[0]))
        elif 'Jside2' in line:
            line_split = line.split('=')
            grid_stats['Jside2'].append(int(line_split[1].split()[0]))
            grid_stats['ILE2'].append(int(line_split[2].split()[0]))
            grid_stats['ITE2'].append(int(line_split[3].split()[0]))

    return grid_stats


def read_bl_data_from_mses(src_file: str):
    r"""
    Reads boundary layer information from ``bl.*`` files generated by MSES

    Parameters
    ==========
    src_file:
      Boundary layer file, usually with the filename ``bl.{airfoil_name}``

    Returns
    =======
    typing.List[dict]
      Boundary layer data list (length :math:`M`) of dictionaries, where :math:`M` is the number of airfoil sides
      (2 times the number of airfoils)
    """
    with open(src_file, 'r') as f:
        line1 = f.readline()
    header_line = line1.replace(' #', '').split()
    header_line.pop(0)
    header_line[-1] = 'Pend'
    df = pd.read_csv(src_file, delim_whitespace=True, skiprows=2, names=header_line)
    bl = [{}]
    side_idx = 0
    # old_x_val = 0
    for idx, s_val in enumerate(df.s):
        if idx == 0:
            for var_name in df.columns:
                bl[side_idx][var_name] = [getattr(df, var_name)[idx]]
        else:
            if s_val == 0.0:  # Criterion for the start of a new airfoil side: arc length = 0
                side_idx += 1
                bl.append({})
                for var_name in df.columns:
                    bl[side_idx][var_name] = [getattr(df, var_name)[idx]]
            else:
                for var_name in df.columns:
                    bl[side_idx][var_name].append(getattr(df, var_name)[idx])
    return bl


def read_field_from_mses(src_file: str):
    r"""
    Reads a field dump file from MSES (by default, of the form ``field.*``) and outputs the information to an array.
    The array has shape :math:`9 \times m \times n`, where :math:`m` is the number of streamlines and :math:`n`
    is the number of streamwise cells in a given streamline.

    The 9 entries in the zeroth axis correspond to the flow variables as follows:
    - 0: :math:`x`
    - 1: :math:`y`
    - 2: :math:`\rho/\rho_\infty` (density)
    - 3: :math:`p/p_\infty` (pressure)
    - 4: :math:`u/V_inf` (:math:`x`-velocity)
    - 5: :math:`v/V_inf` (:math:`y`-velocity)
    - 6: :math:`q/Vinf` (velocity magnitude)
    - 7: :math:`M` (Mach number)
    - 8: :math:`C_p` (pressure coefficient)

    Parameters
    ==========
    src_file: str
        Source file containing the MSES field output

    Returns
    =======
    np.ndarray
        Array of MSES field data, reshaped to :math:`9 \times m \times n`
    """
    data = np.loadtxt(src_file, skiprows=2)
    n_flow_vars = data.shape[1]

    with open(src_file, "r") as f:
        lines = f.readlines()

    n_streamlines = 0
    for line in lines:
        if line == "\n":
            n_streamlines += 1

    n_streamwise_lines = int(data.shape[0] / n_streamlines)

    field_array = np.array([data[:, i].reshape(n_streamlines, n_streamwise_lines).T for i in range(n_flow_vars)])

    return field_array


def read_streamline_grid_from_mses(src_file: str, grid_stats: dict):
    r"""
    Reads the grid of streamlines from an MSES ``grid.*`` file

    Parameters
    ==========
    src_file: str
        File containing the grid output from MPLOT (usually of the form ``grid.*``)

    grid_stats: dict
        Output from the ``read_grid_stats_from_mses()`` function

    Returns
    =======
    typing.Tuple(np.ndarray, np.ndarray)
        Grid x-coordinates and grid y-coordinates split by the stagnation streamlines
    """
    data = np.loadtxt(src_file, skiprows=2)

    with open(src_file, "r") as f:
        lines = f.readlines()

    n_streamlines = 0
    for idx, line in enumerate(lines):
        if line == "\n":
            n_streamlines += 1

    n_streamwise_lines = int(data.shape[0] / n_streamlines)

    x_grid = data[:, 0].reshape(n_streamlines, n_streamwise_lines).T
    y_grid = data[:, 1].reshape(n_streamlines, n_streamwise_lines).T

    streamline_borders = np.sort(grid_stats["Jside2"])
    split_x_grid = np.split(x_grid, streamline_borders, axis=1)
    split_y_grid = np.split(y_grid, streamline_borders, axis=1)

    return split_x_grid, split_y_grid


def convert_blade_file_to_3d_array(src_file: str):
    r"""
    Converts an MSES blade file (by default of the form ``blade.*``) to an array.
    The array has shape :math:`N \times M \times 2`, where :math:`N` is the number of airfoils
    and :math:`M` is the number of discrete airfoil coordinates in a given airfoil.

    Parameters
    ==========
    src_file: str
        Source file containing the MSES blade information (usually ``blade.*``)

    Returns
    =======
    np.ndarray
        Array of the shape :math:`N \times M \times 2`
    """
    blade = np.loadtxt(src_file, skiprows=2)
    airfoil_delimiter_rows = np.argwhere(blade == 999.0)
    array_3d = np.split(blade, np.unique(airfoil_delimiter_rows[:, 0]))
    new_airfoils = [array_3d[0]]
    for airfoil in array_3d[1:]:
        new_airfoil = np.delete(airfoil, 0, axis=0)
        new_airfoils.append(new_airfoil)
    return new_airfoils


if __name__ == '__main__':
    f_ = os.path.join(os.path.dirname(os.getcwd()), 'data', 'test_airfoil', 'mplot.log')
    forces_ = read_forces_from_mses(f_)
    print(f"forces = {forces_}")
