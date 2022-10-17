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
