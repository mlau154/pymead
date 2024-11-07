import os
import re
import typing

import pandas as pd
import numpy as np


flow_var_idx = {"M": 7, "Cp": 8, "p": 3, "rho": 2, "u": 4, "v": 5, "V": 6, "Cpt": 9, "dCpt": 10, "dCp": 11}


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
    y_in_header = False
    header_line_idx = None
    with open(fname, "r") as f:
        for idx, line in enumerate(f):
            if "Cp" not in line:
                continue
            header_line_idx = idx
            if "y" in line:
                y_in_header = True
            break
        else:
            raise ValueError("Could not detect the header line in the XFOIL Cp file")

    # Read in the Cp data for XFOIL versions 6.93 or 6.99 (other versions are untested)
    names = ["x", "y", "Cp"] if y_in_header else ["x", "Cp"]
    cols = {"x": 0, "y": 1, "Cp": 2} if y_in_header else {"x": 0, "Cp": 1}
    df = pd.read_csv(fname, skiprows=header_line_idx + 1, names=names, sep=r"\s+", engine='python')
    array_ = df.to_numpy()
    return {name: array_[:, cols[name]] for name in names}


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
    element_lines = []
    alpha_line, vw_line, fp_line, ad_line, ad_mass_flow_line = None, None, None, None, None
    element = 0
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
        if "mh / rho V" in line:
            ad_mass_flow_line = line
        if "Element" in line:
            element = int(line.split(":")[-1].strip())
        if element != 0 and "CL" in line:
            element_lines.append([line])
        if element != 0 and "CM" in line:
            element_lines[-1].append(line)
        if element != 0 and "top" in line:
            element_lines[-1].append(line)

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

        # Find the non-dimensional mass flow rate due to the actuator disk:
        if ad_mass_flow_line is not None:
            ad_mass_flow_line = ad_mass_flow_line.split("=")
            forces["mh / rho V"] = float(ad_mass_flow_line[-1].strip())
        else:
            forces["mh / rho V"] = 0.0

        # Find the elementwise aerodynamic performance components
        if element_lines:
            for idx, element_line in enumerate(element_lines):
                forces[f"Element-{idx + 1}"] = {
                    "Cl": float(element_line[0].split()[2]),
                    "Cdv": float(element_line[0].split()[5]),
                    "Cm": float(element_line[1].split()[2]),
                    "Cdf": float(element_line[1].split()[5]),
                    "top_xtr": float(element_line[2].split()[3]),
                    "bot_xtr": float(element_line[2].split()[7]),
                }
    except:
        forces = {'Cl': 0.0, 'Cd': 1000.0, 'Cm': 1000.0, 'alf': 0.0, 'Cdv': 1000.0, 'Cdw': 1000.0,
                  'Cdf': 1000.0, 'Cdp': 1000.0, 'Cdh': 1000.0, 'CPK': 1000.0, "mh / rho V": 0.0}

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


def read_actuator_disk_data_mses(mses_log_file: str, grid_stats: dict) -> typing.List[dict]:
    """
    Reads actuator disk data from the MSES log file (usually ``mses.log``).

    Parameters
    ==========
    mses_log_file: str
        Log file generated by MSES after CFD evaluation, usually ``mses.log``.

    grid_stats: dict
        Grid statistics dictionary generated by ``read_grid_stats_from_mses`` in this module.

    Returns
    =======
    typing.List[dict]
        List of actuator disk data, with each element in the list representing a different actuator disk
    """
    with open(mses_log_file, "r") as f:
        lines = f.readlines()

    lines_AD = []
    for line_idx, line in enumerate(lines):
        if "Actuator disk plane" in line:
            lines_AD.append(line_idx)

    data_AD = []
    for line_idx in lines_AD:
        line1, line2, line3 = lines[line_idx].split(), lines[line_idx + 1].split(), lines[line_idx + 2].split()
        data_AD.append({"x_c": float(line1[6]),
                        "i": int(line1[9]),
                        "j_start": int(line1[12]),
                        "j_end": int(line1[14]),
                        "pt2_pt1": float(line2[2]),
                        "dCpo": float(line2[5]),
                        "ht2_ht1": float(line3[2]),
                        "etah": float(line3[5])
                        })

    for data in data_AD:
        flow_section_idx = None
        if data["j_start"] == 1:
            flow_section_idx = 0
        else:
            for j_side_idx, j_side in enumerate(grid_stats["Jside1"][::-1]):
                if data["j_start"] == j_side:
                    flow_section_idx = j_side_idx + 1
        data["flow_section_idx"] = flow_section_idx
        data["field_i_up"] = data["i"] - 2
        data["field_i_down"] = data["i"] - 1
        data["field_j_start"] = data["j_start"] - flow_section_idx - 1
        data["field_j_end"] = data["j_end"] - flow_section_idx - 1

    return data_AD


def read_bl_data_from_mses(src_file: str) -> typing.List[dict]:
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
    df = pd.read_csv(src_file, sep=r"\s+", skiprows=2, names=header_line)
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


def read_Mach_from_mses_file(mses_file: str) -> float:
    """
    Reads the freestream Mach number from the MSES settings file (usually of the form ``mses.*``).

    Parameters
    ==========
    mses_file: str
        Settings file for MSES, most likely of the form ``mses.*``

    Returns
    =======
    float
        Freestream Mach number
    """
    with open(mses_file, "r") as f:
        lines = f.readlines()
    mach_line = lines[2]
    mach_split = mach_line.split()
    mach_float = float(mach_split[0])
    return mach_float


def read_field_from_mses(src_file: str, M_inf: float or None = None, gam: float or None = None):
    r"""
    Reads a field dump file from MSES (by default, of the form ``field.*``) and outputs the information to an array.
    The array has shape :math:`9 \times m \times n` (or :math:`12 \times m \times n` if ``M_inf`` and ``gam`` are
    specified), where :math:`m` is the number of streamlines and :math:`n`
    is the number of streamwise cells in a given streamline.

    The 12 entries in the zeroth axis correspond to the flow variables as follows:

    * 0: :math:`x`
    * 1: :math:`y`
    * 2: :math:`\rho/\rho_\infty` (density)
    * 3: :math:`p/p_\infty` (pressure)
    * 4: :math:`u/V_\infty` (:math:`x`-velocity)
    * 5: :math:`v/V_\infty` (:math:`y`-velocity)
    * 6: :math:`|V|/V_\infty` (velocity magnitude)
    * 7: :math:`M` (Mach number)
    * 8: :math:`C_p` (pressure coefficient)
    * 9: :math:`C_{p_t}` (total pressure coefficient)
    * 10: :math:`\Delta C_{p_t}` (change in total pressure coefficient relative to the previous streamwise cell)
    * 11: :math:`\Delta C_p` (change in pressure coefficient relative to the previous streamwise cell)

    Note that for :math:`\Delta C_{p_t}` and :math:`\Delta C_p`, the value of the first streamwise cells (the cells
    along the inlet plane) is defined to be 0. Also note that entries 9, 10, and 11 are only available if ``M_inf``
    and ``gam`` are specified.

    Parameters
    ==========
    src_file: str
        Source file containing the MSES field output

    M_inf: float or None
        Freestream Mach number (must be set if entries 9, 10, or 11 are needed). Default: ``None``.

    gam: float or None
        Specific heat ratio (must be set if entries 9, 10, or 11 are needed). Default: ``None``.

    Returns
    =======
    np.ndarray
        Array of MSES field data, reshaped to :math:`9 \times m \times n`, or :math:`12 \times m \times n` if
        ``M_inf`` and ``gam`` are specified.
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

    if M_inf is not None and gam is not None:
        Cpt = field_array[3][:, :] * (1 + (gam - 1) / 2 * field_array[7][:, :] ** 2) ** (gam / (gam - 1)) * (2 / gam / M_inf**2)
        Cp = field_array[flow_var_idx["Cp"]][:, :]
        Cpt = np.array([Cpt])
        Cp = np.array([Cp])
        field_array = np.concatenate((field_array, Cpt))
        dCpt = np.zeros(Cpt[0].shape)
        dCp = np.zeros(Cpt[0].shape)
        for idx in range(1, Cpt[0].shape[0]):
            dCpt[idx, :] = Cpt[0][idx, :] - Cpt[0][idx - 1, :]
            dCp[idx, :] = Cp[0][idx, :] - Cp[0][idx - 1, :]
        dCpt = np.array([dCpt])
        dCp = np.array([dCp])
        field_array = np.concatenate((field_array, dCpt))
        field_array = np.concatenate((field_array, dCp))

    return field_array


def read_field_variables_names_mses(field_file: str) -> typing.List[str]:
    with open(field_file, "r") as src_file:
        line = src_file.readline()
    headers = line.replace("#", "").replace("\n", "").split()
    return headers


def export_mses_field_to_tecplot_ascii(output_file: str, field_file: str, grid_stats_file: str,
                                       grid_file: str, **kwargs):

    field_array = read_field_from_mses(field_file, **kwargs)
    grid_stats = read_grid_stats_from_mses(grid_stats_file)
    x_grid, y_grid = read_streamline_grid_from_mses(grid_file, grid_stats)
    headers = read_field_variables_names_mses(field_file)
    with open(output_file, "w") as ascii_file:
        ascii_file.write(f'VARIABLES = ')
        for header_idx, header in enumerate(headers):
            if header_idx < len(headers) - 1:
                ascii_file.write(f'"{header}", ')
            else:
                ascii_file.write(f'"{header}"\n')
        starting_streamline = 0
        for zone_idx in range(len(x_grid)):
            i_max, j_max = x_grid[zone_idx].shape[0], x_grid[zone_idx].shape[1]
            ascii_file.write(f'ZONE T="ZONE {zone_idx + 1}", I={x_grid[zone_idx].shape[0]}, J={x_grid[zone_idx].shape[1]}, DATAPACKING=BLOCK, VARLOCATION=(')
            entry_counter = 5  # Used to limit the line length less than 4000 characters, as required by Tecplot
            for header_idx in range(2, len(headers)):
                if header_idx < len(headers) - 1:
                    if entry_counter < 6:
                        ascii_file.write(f"{header_idx + 1}=CELLCENTERED, ")
                    else:
                        ascii_file.write(f"{header_idx + 1}=CELLCENTERED,\n")
                        entry_counter = 0
                    entry_counter += 1
                else:
                    ascii_file.write(f"{header_idx + 1}=CELLCENTERED)\n")

            # Write the x values (node-centered)
            entry_counter = 0  # Used to limit the line length less than 4000 characters, as required by Tecplot
            for j in range(j_max):
                for i in range(i_max):
                    if entry_counter < 10:
                        ascii_file.write(str(x_grid[zone_idx][i, j]) + " ")
                        entry_counter += 1
                    else:
                        ascii_file.write(str(x_grid[zone_idx][i, j]) + "\n")
                        entry_counter = 0
            ascii_file.write("\n")

            # Write the y values (node-centered)
            entry_counter = 0  # Used to limit the line length less than 4000 characters, as required by Tecplot
            for j in range(j_max):
                for i in range(i_max):
                    if entry_counter < 10:
                        ascii_file.write(str(y_grid[zone_idx][i, j]) + " ")
                        entry_counter += 1
                    else:
                        ascii_file.write(str(y_grid[zone_idx][i, j]) + "\n")
                        entry_counter = 0
            ascii_file.write("\n")

            # Write the other variables (cell-centered)
            entry_counter = 0  # Used to limit the line length less than 4000 characters, as required by Tecplot
            for header_idx in range(2, len(headers)):
                for j in range(starting_streamline, j_max + starting_streamline - 1):
                    for i in range(0, i_max - 1):
                        if entry_counter < 10:
                            ascii_file.write(str(field_array[header_idx, i, j]) + " ")
                            entry_counter += 1
                        else:
                            ascii_file.write(str(field_array[header_idx, i, j]) + "\n")
                            entry_counter = 0
                ascii_file.write("\n")
            starting_streamline += x_grid[zone_idx].shape[1] - 1
    pass


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


def read_polar(airfoil_name: str, base_dir: str) -> typing.Dict[str, typing.List[float]]:
    """
    Reads the ``polar.xxx`` file produced by MPOLAR and converts the data to a dictionary of lists

    Parameters
    ----------
    airfoil_name: str
        Name of the airfoil provided to MSES (also the name of the sub-folder inside ``base_dir`` containing the
        analysis files)

    base_dir: str
        Base directory of the analysis. The file being read should have the form
        ``base_dir/airfoil_name/polar.airfoil_name``

    Returns
    -------
    typing.Dict[str, typing.List[float]]
        Dictionary where each key is a string corresponding to an aerodynamic performance variable and each
        value is a list of the evaluation of that performance variable at every point along the polar.
        For example, the dictionary might look something like ``{"alf": [-1.0, 0.0, 1.0], "Cl": [0.2, 0.3, 0.4], ...}``
    """
    data = np.loadtxt(os.path.join(base_dir, airfoil_name, f"polar.{airfoil_name}"), skiprows=13)
    keys = ["alf", "Cl", "Cd", "Cm", "Cdw", "Cdv", "Cdp", "Ma", "Re", "top_xtr", "bot_xtr"]
    return {k: v.tolist() for (k, v) in zip(keys, data.T)}


if __name__ == "__main__":
    my_folder = r"C:\Users\mlauer2\Documents\dissertation\data\MSESRuns\OptUnderwingFreeTransition"
    export_mses_field_to_tecplot_ascii(
        os.path.join(my_folder, "OptUnderwingFreeTransition.dat"),
        os.path.join(my_folder, "field.OptUnderwingFreeTransition"),
        os.path.join(my_folder, "mplot_grid_stats.log"),
        os.path.join(my_folder, "grid.OptUnderwingFreeTransition")
    )
