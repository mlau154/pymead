from functools import reduce
from copy import deepcopy
import numpy as np


def set_all_dict_values(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            set_all_dict_values(v)
        else:
            v.param_dict = d


def assign_airfoil_tags_to_param_dict(d: dict, airfoil_tag: str):
    for k, v in d.items():
        if isinstance(v, dict):
            assign_airfoil_tags_to_param_dict(v, airfoil_tag)
        else:
            v.airfoil_tag = airfoil_tag


def assign_names_to_params_in_param_dict(d: dict, name_str: str = ''):
    for k, v in d.items():
        if isinstance(v, dict):
            if len(v) > 0:
                name_str += f'{k}.'
            assign_names_to_params_in_param_dict(v, name_str)
        else:
            v.name = name_str + k


def recursive_get(d, *keys):
    """From answer by Thomas Orozco (https://stackoverflow.com/a/28225747)"""
    return reduce(lambda c, k: c.get(k, {}), keys, d)


def unravel_param_dict(d: dict, output_dict: dict, prep_for_json: bool = True):
    for k, v in d.items():
        if isinstance(v, dict):
            output_dict[k] = {}
            unravel_param_dict(v, output_dict[k], prep_for_json=prep_for_json)
        else:
            param_dict_attrs = vars(v)
            for attr_key, attr in param_dict_attrs.items():
                if prep_for_json:
                    if isinstance(attr, np.ndarray):
                        param_dict_attrs[attr_key] = attr.tolist()
                    if attr_key in ['mea', 'param_dict', 'free_point', 'anchor_point', 'affects', 'depends_on',
                                    'function_dict', 'func']:
                        param_dict_attrs[attr_key] = None
            output_dict[k] = param_dict_attrs


def unravel_param_dict_deepcopy(d: dict, output_dict: dict, prep_for_json: bool = True):
    for k, v in d.items():
        if isinstance(v, dict):
            output_dict[k] = {}
            unravel_param_dict_deepcopy(v, output_dict[k], prep_for_json=prep_for_json)
        else:
            param_dict_attrs = vars(v)
            temp_dict = {}
            for attr_key, attr in param_dict_attrs.items():
                if prep_for_json:
                    if isinstance(attr, np.ndarray):
                        temp_dict[attr_key] = deepcopy(attr.tolist())
                    if attr_key in ['mea', 'param_dict', 'free_point', 'anchor_point', 'affects', 'depends_on',
                                    'function_dict', 'func']:
                        temp_dict[attr_key] = None
                    else:
                        if not isinstance(attr, np.ndarray):
                            temp_dict[attr_key] = deepcopy(attr)
            output_dict[k] = temp_dict
