

def calculate_normal_shock_total_pressure_ratio(M_up: float, gam: float) -> float:
    """
    Normal-shock relation that calculates the total pressure ratio across a shock (:math:`p_{t,y} / p_{t,x}` ,
    where :math:`x` is the state immediately upstream of the shock, and :math:`y` is the state immediately downstream
    of the shock).

    Parameters
    ==========
    M_up: float
        Mach number immediately upstream of the shock and normal to the shock (:math:`M_x`)

    gam: float
        Specific heat ratio

    Returns
    =======
    float
        Total pressure ratio across the shock wave, :math:`p_{t,y} / p_{t,x}`
    """
    A = (gam + 1) / 2 * M_up**2
    B = 1 + (gam - 1) / 2 * M_up**2
    C = 2 * gam * M_up**2 / (gam + 1)
    D = (gam - 1) / (gam + 1)
    return (A / B) ** (gam / (gam - 1)) * (C - D) ** (1 / (1 - gam))
