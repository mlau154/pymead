import json
import os
import pickle
import typing


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
            f.write("\b\n")
