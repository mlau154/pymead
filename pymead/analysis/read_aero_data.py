import pandas as pd
import os
import time


def read_Cl_from_file_panel_fort(f: str):
    """
    Reads the lift coefficient from the 'LIFT.DAT' file output by panel_fort
    """
    with open(f, 'r') as Cl_file:
        line = Cl_file.readline()
    str_Cl = ''
    for ch in line:
        if ch.isdigit() or ch in ['.', 'e', 'E', '-']:
            str_Cl += ch
    return float(str_Cl)


def read_Cp_from_file_panel_fort(f: str):
    """
    Reads the Cp data from the format output by panel_fort and converts it to a numpy array
    """
    df = pd.read_csv(f, names=['x/c', 'Cp'])  # names just added here to make sure the first line is not treated as a
    # header
    array_ = df.to_numpy()
    return {'x': array_[0, :], 'Cp': array_[1, :]}


def read_Cp_from_file_xfoil(f: str):
    """
    Reads the Cp data from the format output by XFOIL and converts it to a numpy array
    """
    df = pd.read_csv(f, skiprows=3, names=['x', 'y', 'Cp'], sep='\s+', engine='python')
    array_ = df.to_numpy()
    return {'x': array_[:, 0], 'y': array_[:, 1], 'Cp': array_[:, 2]}


def read_aero_data_from_xfoil(fpath: str, aero_data: dict):
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
    # print(f"line1 = {line1}")
    return line1, line2


def read_reverse_order(file_name):
    """
    From https://thispointer.com/python-read-a-file-in-reverse-order-line-by-line/
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
