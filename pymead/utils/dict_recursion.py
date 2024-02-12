from functools import reduce
from copy import deepcopy
import numpy as np
import benedict


class DictValueNotEqualException(Exception):
    """
    Exception used to break out of the dictionary comparison loop immediately if two values do not match
    """
    pass


def compare_dicts_floating_precision(dict_1: dict, dict_2: dict, atol: float) -> bool:
    """
    Compares two dictionaries recursively. Early return ``False`` if any nested dictionary is found to have a different
    length, and keys are found not to match, any float values found to not match by the floating-point precision
    specified by ``atol``, or any non-float values found not to match.

    Parameters
    ----------
    dict_1: dict
        First dictionary to compare

    dict_2: dict
        Second dictionary to compare

    atol: float
        Floating-point precision used to compare float values

    Returns
    -------
    bool
        ``True`` if the dictionaries match, ``False`` otherwise
    """
    def compare(d1: dict, d2: dict):
        for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
            if k1 != k2:
                # Found non-matching keys
                raise DictValueNotEqualException
            if isinstance(v1, dict) and isinstance(v2, dict):
                if len(v1) != len(v2):
                    # Found dictionaries of different length
                    raise DictValueNotEqualException
                compare(v1, v2)
            else:
                if isinstance(v1, float) and isinstance(v2, float):
                    if abs(v1 - v2) > atol:
                        # Found float values not equal
                        raise DictValueNotEqualException
                else:
                    if v1 != v2:
                        # Found non-float values not equal
                        raise DictValueNotEqualException

    try:
        compare(dict_1, dict_2)
        return True
    except DictValueNotEqualException:
        return False


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


def assign_names_to_params_in_param_dict(d: dict):
    dben = benedict.benedict(d)
    keypaths = dben.keypaths()
    for k in keypaths:
        p = dben[k]
        if not isinstance(p, dict):
            p.name = k


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


def unravel_param_dict_deepcopy(d: dict, output_dict: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            output_dict[k] = {}
            unravel_param_dict_deepcopy(v, output_dict[k])
        else:
            param_dict_attrs = vars(v)
            temp_dict = {}
            for attr_key, attr in param_dict_attrs.items():
                if attr_key in ['name', '_value', 'active', 'linked', 'func_str', 'bounds']:
                    if isinstance(attr, np.ndarray):
                        temp_dict[attr_key] = deepcopy(attr.tolist())
                    else:
                        temp_dict[attr_key] = deepcopy(attr)
            output_dict[k] = temp_dict
