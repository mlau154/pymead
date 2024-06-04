"""
Templates to use for aerodynamic forces output of XFOIL and MSES
"""


XFOIL = {'Cd': 0.01, 'Cl': 0.5, 'alf': 0.0, 'Cm': 0.01, 'Cdf': 0.005, 'Cdp': 0.005, 'L/D': 50.0, 'Cp': {}}
MSES = {'Cd': 0.01, 'Cl': 0.5, 'alf': 0.0, 'Cm': 0.01, 'Cdf': 0.005, 'Cdp': 0.005, 'Cdw': 0.005,
        'Cdv': 0.005, 'Cdh': 0.0, 'L/D': 50.0, 'CPK': 0.1, 'BL': []}
XFOIL_MULTIPOINT = {k: [v] * 100 for k, v in XFOIL.items()}
MSES_MULTIPOINT = {k: [v] * 100 for k, v in MSES.items()}
XFOIL_BLANK = {k: [] for k in XFOIL.keys()}
MSES_BLANK = {k: [] for k in MSES.keys()}
