import json
import os
import pickle
import dill


def save_data(var, file):
    if os.path.splitext(file)[-1] == '.pkl':
        with open(file, 'wb') as file:
            pickle.dump(var, file, protocol=-1)
    elif os.path.splitext(file)[-1] == '.dill' or os.path.splitext(file)[-1] == '.mead':
        with open(file, 'wb') as file:
            dill.dump(var, file)
    elif os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'w') as file:
            json.dump(var, file, indent=4)
    else:
        raise Exception('Invalid file extension for data save! Current available choices: .pkl, .dill, .json, .jmea')


def load_data(file):
    if os.path.splitext(file)[-1] == '.pkl':
        with open(file, 'rb') as file:
            var = pickle.load(file)
        return var
    elif os.path.splitext(file)[-1] == '.dill' or os.path.splitext(file)[-1] == '.mead':
        with open(file, 'rb') as file:
            var = dill.load(file)
        return var
    elif os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'r') as file:
            var = json.load(file)
        return var
    else:
        raise Exception('Invalid file extension for data load! Current available choices: .pkl, .dill, .json, .jmea')
