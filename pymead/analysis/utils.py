

def viscosity_calculator(T: float, rho: float or None = None, input_units: str = 'K') -> float:
    r"""
    Used to calculate the viscosity from temperature using Sutherland's law for dialogs in the GUI

    Parameters
    ==========
    T: float
        The input temperature (can be in units of :math:`K`, :math:`^{\circ}C`, or :math:`^{\circ}F`)

    rho: float or None
      The density, specified if an output of kinematic viscosity is desired instead of dynamic viscosity.
      Default: ``None``.

    input_units: str
      Input units. Must be one of ``"K"``, ``"C"``, or ``"F"``. Default: ``"K"``

    Returns
    =======
    float
        The dynamic viscosity if the density is not specified, or the kinematic viscosity if the density *is* specified
    """
    if input_units == 'K':
        pass
    elif input_units == 'C':
        T = T + 273.15  # convert from Celsius to Kelvin
    elif input_units == 'F':
        T = (T - 32) * 5 / 9 + 273.15  # convert from Fahrenheit to Kelvin
    else:
        raise ValueError('Invalid selection for \'input_units\' (must be one of \'K\', \'C\', or \'F\')')
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    mu = mu_ref * (T / T_ref) ** (3 / 2) * (T_ref + S) / (T + S)
    if rho is None:
        return mu
    else:
        nu = mu / rho
        return nu
