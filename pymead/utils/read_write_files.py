import json
import os
import pickle
import typing

from PyQt5.QtCore import QStandardPaths

from pymead import q_settings


def save_data(var, file):
    if os.path.splitext(file)[-1] == '.pkl':
        with open(file, 'wb') as file:
            pickle.dump(var, file, protocol=-1)
    elif os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'w') as file:
            json.dump(var, file, indent=4)
    else:
        raise Exception('Invalid file extension for data save! Current available choices: .pkl, .json, .jmea')


def load_data(file):
    if os.path.splitext(file)[-1] == '.pkl':
        with open(file, 'rb') as file:
            var = pickle.load(file)
        return var
    elif os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'r') as file:
            var = json.load(file)
        return var
    else:
        raise Exception('Invalid file extension for data load! Current available choices: .pkl, .json, .jmea')


def write_tuple_tuple_to_file(fname: str, data: typing.Tuple[tuple]):
    """Data must be 2-D"""
    with open(fname, 'w') as f:
        for row in data:
            for col in row:
                f.write(f"{col} ")
            f.write("\n")


def load_documents_path(settings_var: str):
    """
    Utility function that returns a specified QSettings path location if saved, otherwise returns the Documents location

    Parameters
    ==========
    settings_var: str
        Key to the path location in ``pymead.q_settings``

    Returns
    =======
    str
        Path specified by ``settings_var``, otherwise the Documents location
    """
    if q_settings.contains(settings_var):
        path = q_settings.value(settings_var)
    else:
        path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return path
