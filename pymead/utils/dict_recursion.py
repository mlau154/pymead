def set_all_dict_values(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            set_all_dict_values(v)
        else:
            v.param_dict = d
