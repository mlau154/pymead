from functools import reduce


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
