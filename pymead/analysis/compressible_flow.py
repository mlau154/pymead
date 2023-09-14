

def calculate_normal_shock_total_pressure_ratio(M_up: float, gam: float):
    A = (gam + 1) / 2 * M_up**2
    B = 1 + (gam - 1) / 2 * M_up**2
    C = 2 * gam * M_up**2 / (gam + 1)
    D = (gam - 1) / (gam + 1)
    return (A / B) ** (gam / (gam - 1)) * (C - D) ** (1 / (1 - gam))
