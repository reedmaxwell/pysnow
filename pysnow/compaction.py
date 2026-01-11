"""
Snow compaction and density evolution.

Available methods:
- 'anderson': Anderson (1976) with metamorphism - CLM style
- 'essery': Essery et al. (2013) with viscosity
- 'simple': Simple overburden compaction
"""

import numpy as np
from .constants import TFRZ, RHO_ICE, GRAVITY


def calc_new_snow_density(t_air, method='anderson'):
    """
    Calculate fresh snow density.

    Parameters
    ----------
    t_air : float
        Air temperature [K]

    Returns
    -------
    density : float
        Fresh snow density [kg/m³]
    """
    t_c = t_air - TFRZ

    if method == 'anderson':
        # Anderson (1976) / CLM formula
        if t_c > 2.0:
            density = 169.15
        elif t_c > -15.0:
            density = 50.0 + 1.7 * (t_c + 15.0) ** 1.5
        else:
            density = 50.0
    else:
        # Simple linear
        density = max(50, min(200, 100 + 5 * t_c))

    return density


def update_density(density, swe, depth, t_snow, dt, method='essery', **kwargs):
    """
    Update snow density due to compaction.

    Parameters
    ----------
    density : float
        Current density [kg/m³]
    swe : float
        Snow water equivalent [mm]
    depth : float
        Snow depth [m]
    t_snow : float
        Snow temperature [K]
    dt : float
        Timestep [s]
    method : str
        Compaction method

    Returns
    -------
    density : float
        Updated density [kg/m³]
    depth : float
        Updated depth [m]
    """
    if swe <= 0 or depth <= 0:
        return density, depth

    if method == 'essery':
        return _essery_compaction(density, swe, depth, t_snow, dt, **kwargs)
    elif method == 'anderson':
        return _anderson_compaction(density, swe, depth, t_snow, dt, **kwargs)
    elif method == 'simple':
        return _simple_compaction(density, swe, depth, t_snow, dt, **kwargs)
    else:
        raise ValueError(f"Unknown compaction method: {method}")


def _essery_compaction(density, swe, depth, t_snow, dt,
                       c1=2.8e-6, c2=0.042, c3=0.046, c4=0.081, c5=0.018,
                       rho_0=150.0, eta_0=3.7e7, rho_max=550.0, **kwargs):
    """
    Essery et al. (2013) viscosity-based compaction.

    Parameters
    ----------
    density : float
        Current density [kg/m³]
    swe : float
        Snow water equivalent [mm]
    depth : float
        Snow depth [m]
    t_snow : float
        Snow temperature [K]
    dt : float
        Timestep [s]
    c1-c5 : float
        Compaction parameters
    rho_0 : float
        Reference density [kg/m³]
    eta_0 : float
        Reference viscosity [kg/(m·s)]
    rho_max : float
        Maximum density [kg/m³]
    """
    t_c = t_snow - TFRZ
    t_neg = max(0, -t_c)  # Positive value for cold temps

    # Viscosity (increases with cold and density)
    eta = eta_0 * np.exp(c4 * t_neg + c5 * density)

    # Overburden mass (assume half the pack is above)
    mass = swe / 2.0  # kg/m² = mm

    # Overburden compaction
    d_rho_overburden = density * mass * GRAVITY / eta

    # Metamorphic compaction (temperature-dependent)
    d_rho_metamorphism = c1 * np.exp(-c2 * t_neg - c3 * max(0, density - rho_0))

    # Total density change rate
    d_rho = density * (d_rho_overburden + d_rho_metamorphism)

    # Update density
    new_density = density + d_rho * dt
    new_density = min(rho_max, new_density)

    # Update depth (conserve SWE)
    # depth [m] = SWE [kg/m²] / density [kg/m³]
    new_depth = swe / new_density

    return new_density, new_depth


def _anderson_compaction(density, swe, depth, t_snow, dt,
                         c1=0.01, c2=21.0, rho_max=550.0, **kwargs):
    """
    Anderson (1976) compaction used in CLM.
    Simpler than Essery.
    """
    t_c = t_snow - TFRZ

    # Compaction rate decreases with density
    cr = c1 * np.exp(-c2 * (density / 1000.0))

    # Temperature factor (slower when cold)
    if t_c < 0:
        tf = np.exp(0.08 * t_c)
    else:
        tf = 1.0

    # Overburden factor
    of = 1.0 + swe / 100.0

    # Density change
    d_rho = density * cr * tf * of * dt / 3600.0

    new_density = min(rho_max, density + d_rho)
    new_depth = swe / new_density

    return new_density, new_depth


def _simple_compaction(density, swe, depth, t_snow, dt,
                       rate=0.001, rho_max=500.0, **kwargs):
    """
    Simple time-based compaction.
    """
    # Linear approach to maximum
    d_rho = rate * (rho_max - density) * dt / 3600.0
    new_density = min(rho_max, density + d_rho)
    new_depth = swe / new_density

    return new_density, new_depth


def add_fresh_snow(density_old, swe_old, depth_old, snowfall, density_new):
    """
    Add fresh snow to pack and compute new bulk properties.

    Parameters
    ----------
    density_old : float
        Existing pack density [kg/m³]
    swe_old : float
        Existing SWE [mm]
    depth_old : float
        Existing depth [m]
    snowfall : float
        New snowfall [mm]
    density_new : float
        Fresh snow density [kg/m³]

    Returns
    -------
    density : float
        New bulk density [kg/m³]
    swe : float
        New SWE [mm]
    depth : float
        New depth [m]
    """
    if snowfall <= 0:
        return density_old, swe_old, depth_old

    # New snow depth: depth [m] = SWE [kg/m²] / density [kg/m³]
    depth_new = snowfall / density_new

    # Total SWE and depth
    swe = swe_old + snowfall
    depth = depth_old + depth_new

    # Weighted harmonic mean density (Essery eq. 17)
    if swe > 0:
        density = swe / (swe_old / max(1, density_old) + snowfall / density_new)
    else:
        density = density_new

    return density, swe, depth
