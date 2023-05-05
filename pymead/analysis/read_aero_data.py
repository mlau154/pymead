import pandas as pd
import os
import re
import typing


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
    df = pd.read_csv(fname, skiprows=3, names=['x', 'y', 'Cp'], sep='\s+', engine='python')
    array_ = df.to_numpy()
    return {'x': array_[:, 0], 'y': array_[:, 1], 'Cp': array_[:, 2]}


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
                  'Cdf': 1000.0, 'Cdp': 1000.0, 'Cdh': 1000.0}

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
            print(f"{[idx for idx in line_split.split()] = }")
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


if __name__ == '__main__':
    f_ = os.path.join(os.path.dirname(os.getcwd()), 'data', 'test_airfoil', 'mplot.log')
    forces_ = read_forces_from_mses(f_)
    print(f"forces = {forces_}")
