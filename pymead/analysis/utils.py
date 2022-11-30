

def viscosity_calculator(T, rho=None, input_units: str = 'K'):
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
