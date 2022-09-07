import numpy as np
from pymead.core.param import Param


class ParamSetup:
    def __init__(self, generate_unlinked_param_dict, generate_linked_param_dict, *args, **kwargs):
        """
        ### Description:

        This class sets up the parameter dictionaries and prepares them for extraction or override.

        ### Args:

        `generate_unlinked_param_dict`: A method that returns a parameter `dict` of unlinked parameters
        (that is, `pymead.core.param.Param`s with `linked=False`)

        `generate_linked_param_dict`: A method that returns a parameter `dict` of linked parameters (that is `pymead.core.param.Param`s with `linked=True`). Takes the unlinked parameter dictionary as input and must return the modified parameter dictionary which includes the linked parameter dictionary. This method can also simply do nothing but return the input parameter dictionary.
        """
        self.generate_unlinked_param_dict = generate_unlinked_param_dict
        self.generate_linked_param_dict = generate_linked_param_dict
        self.param_dict = None
        self.active_unlinked_params = None
        self.parameter_info = None
        self.generate_param_dict(*args, **kwargs)
        self.extract_parameters()

    def generate_param_dict(self, *args, **kwargs):
        """
        ### Description:

        Generates a parameter dictionary from the `generate_unlinked_param_dict` and `generate_linked_param_dict`
        input methods. Raises an exception if the return type is not a `dict`.
        """
        param_dict = self.generate_unlinked_param_dict(*args, **kwargs)
        param_dict = self.generate_linked_param_dict(param_dict, *args, **kwargs)
        if not isinstance(param_dict, dict):
            raise Exception(f'ParamSetup._generate_param_dict must return a dictionary. Return type was '
                            f'{type(param_dict)}')
        else:
            self.param_dict = param_dict
            self.extract_parameters()

    def extract_parameters(self):
        """
        ### Description:

        This function extracts every parameter from the `param_dict` with `active=True` and `linked=False` as a `dict`
        of parameter information.

        ### Returns:

        The list of parameters and a dictionary contain parameter information
        """
        if isinstance(self.param_dict, dict):
            for key, value in self.param_dict.items():
                if isinstance(value, Param):
                    value.name = key
            self.active_unlinked_params = [param for param in self.param_dict.values()
                                           if isinstance(param, Param) and param.active and not param.linked]
        else:
            raise TypeError('Invalid type input for extra_parameters. Must be a list or dictionary of Params.')
        self.parameter_info = {
            'values': [param.value for param in self.active_unlinked_params],
            'bounds_normalized_values': [np.divide(param.value - param.bounds[0], param.bounds[1] - param.bounds[0])
                                         for param in self.active_unlinked_params],
            'bounds': [param.bounds for param in self.active_unlinked_params],
            'names': [param.name for param in self.active_unlinked_params],
            'units': [param.units for param in self.active_unlinked_params],
            'scale_value': [param.scale_value for param in self.active_unlinked_params],
            'n_params': len(self.active_unlinked_params),
        }
        return self.active_unlinked_params, self.parameter_info

    def override_parameters(self, parameter_info_values: list, normalized: bool = False, *args, **kwargs):
        """
        ### Description:

        Overrides the parameters in the parameter dictionary using the input `list` of values. If `normalized` is true,
        this method will convert the input values back into non-normalized values.

        ### Args:

        `parameter_info_values`: A `list` of values with which to override `pymead.core.param.Param`s.

        `normalized`: A `bool` stating whether the input values are normalized by the `pymead.core.param.Param`
        `bounds`

        ### Returns:

        The parameter dictionary modified by the override parameters
        """
        for idx, name in enumerate(self.parameter_info['names']):
            if normalized:
                self.param_dict[name].value = np.multiply(parameter_info_values[idx],
                                                          self.parameter_info['bounds'][idx][1] -
                                                          self.parameter_info['bounds'][idx][0]) + \
                                              self.parameter_info['bounds'][idx][0]
            else:
                self.param_dict[name].value = parameter_info_values[idx]
        self.param_dict = self.generate_linked_param_dict(self.param_dict, *args, **kwargs)
        self.extract_parameters()
        return self.param_dict
