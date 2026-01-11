"""
Snow albedo schemes.

Available methods:
- 'constant': Fixed albedo value
- 'clm_decay': Simple exponential decay with age (CLM-style)
- 'vic': VIC model with separate cold/thaw coefficients
- 'essery': Essery et al. (2013) with new snow refresh
- 'tarboton': UEB model temperature-dependent aging
- 'snicar_lite': Simplified SNICAR with grain size evolution
"""

import numpy as np
from .constants import TFRZ


def calc_albedo(method='vic', **kwargs):
    """
    Calculate snow albedo.

    Parameters
    ----------
    method : str
        Albedo calculation method
    **kwargs : dict
        Method-specific parameters including:
        - age: snow age [seconds]
        - t_snow: snow temperature [K]
        - snowfall: fresh snowfall rate [mm/hr]
        - grain_radius: effective grain radius [μm] (for snicar_lite)
        - albedo_prev: previous timestep albedo
        - dt: timestep [seconds]

    Returns
    -------
    albedo : float
        Snow albedo [0-1]
    grain_radius : float or None
        Updated grain radius [μm] (only for snicar_lite)
    """

    if method == 'constant':
        return _constant(**kwargs), None
    elif method == 'clm_decay':
        return _clm_decay(**kwargs), None
    elif method == 'vic':
        return _vic(**kwargs), None
    elif method == 'essery':
        return _essery(**kwargs), None
    elif method == 'tarboton':
        return _tarboton(**kwargs), None
    elif method == 'snicar_lite':
        return _snicar_lite(**kwargs)
    else:
        raise ValueError(f"Unknown albedo method: {method}")


def _constant(albedo=0.8, **kwargs):
    """Fixed albedo value."""
    return albedo


def _clm_decay(age, albedo_max=0.85, conn=0.2, cons=0.05, tau=86400.0, **kwargs):
    """
    CLM-style exponential decay with age.

    Parameters
    ----------
    age : float
        Snow age [seconds]
    albedo_max : float
        Maximum fresh snow albedo
    conn : float
        VIS decay coefficient
    cons : float
        NIR decay coefficient (unused in broadband)
    tau : float
        Decay timescale [seconds]
    """
    # Broadband approximation (average VIS and NIR)
    decay = 1.0 - np.exp(-age / tau)
    albedo = albedo_max - conn * decay

    return max(0.5, albedo)


def _vic(age, t_snow, albedo_new=0.85, albedo_min=0.5,
         accum_a=0.94, accum_b=0.58, thaw_a=0.82, thaw_b=0.46, **kwargs):
    """
    VIC model with separate cold (accumulating) and warm (thawing) decay.

    Key feature: Faster decay during melt conditions.

    Parameters
    ----------
    age : float
        Snow age [seconds]
    t_snow : float
        Snow surface temperature [K]
    albedo_new : float
        Fresh snow albedo
    albedo_min : float
        Minimum albedo
    accum_a, accum_b : float
        Cold/accumulating decay parameters
    thaw_a, thaw_b : float
        Warm/thawing decay parameters
    """
    age_days = age / 86400.0
    age_days = max(0.001, age_days)  # Avoid zero

    if t_snow < TFRZ:
        # Cold/accumulating conditions - slow decay
        albedo = albedo_new * (accum_a ** (age_days ** accum_b))
    else:
        # Thawing conditions - fast decay
        albedo = albedo_new * (thaw_a ** (age_days ** thaw_b))

    return max(albedo_min, albedo)


def _essery(age, t_snow, snowfall, dt, albedo_prev,
            albedo_max=0.85, albedo_min=0.5,
            s_o=10.0, tau_a=1e7, tau_m=3.6e5, t_melt=-0.5, **kwargs):
    """
    Essery et al. (2013) Option 1 with new snow refresh.

    Parameters
    ----------
    age : float
        Snow age [seconds]
    t_snow : float
        Snow temperature [K]
    snowfall : float
        Snowfall in current timestep [mm]
    dt : float
        Timestep [seconds]
    albedo_prev : float
        Previous timestep albedo
    albedo_max : float
        Maximum albedo
    albedo_min : float
        Minimum albedo
    s_o : float
        Snowfall for full refresh [kg/m²]
    tau_a : float
        Cold snow aging timescale [s]
    tau_m : float
        Melting snow aging timescale [s]
    t_melt : float
        Melt temperature threshold [°C]
    """
    t_c = t_snow - TFRZ

    # Start with previous albedo
    albedo = albedo_prev

    # New snow refresh
    if snowfall > 0:
        refresh = min(1.0, snowfall / s_o)
        albedo = albedo + (albedo_max - albedo) * refresh

    # Age-based decay
    if t_c < t_melt:
        # Cold snow - slow linear decay
        albedo = albedo - dt / tau_a
    else:
        # Melting snow - exponential decay toward minimum
        albedo = (albedo - albedo_min) * np.exp(-dt / tau_m) + albedo_min

    return np.clip(albedo, albedo_min, albedo_max)


def _tarboton(age, t_snow, albedo_vis_new=0.95, albedo_ir_new=0.65,
              c_vis=0.2, c_ir=0.5, **kwargs):
    """
    Tarboton/UEB model with temperature-dependent aging.

    Parameters
    ----------
    age : float
        Snow age [seconds]
    t_snow : float
        Snow temperature [K]
    """
    t_k = max(t_snow, 200.0)  # Avoid extreme values

    # Temperature-dependent aging rate
    # Faster aging near melting point
    r = np.exp(5000.0 * (1.0 / 273.16 - 1.0 / t_k))

    # Age factor
    age_days = age / 86400.0
    fage = age_days * r / (1.0 + age_days * r)

    # Broadband approximation
    albedo_vis = albedo_vis_new * (1.0 - c_vis * fage)
    albedo_ir = albedo_ir_new * (1.0 - c_ir * fage)

    # Simple average (could weight by solar spectrum)
    albedo = 0.5 * albedo_vis + 0.5 * albedo_ir

    return max(0.4, albedo)


def _snicar_lite(t_snow, snowfall, swe, dt, grain_radius=None,
                 r_min=50.0, r_max=1500.0, dr_dry=1.0, dr_wet=50.0,
                 albedo_ref=0.85, **kwargs):
    """
    Simplified SNICAR with grain size evolution.

    Tracks effective grain radius which affects albedo.

    Parameters
    ----------
    t_snow : float
        Snow temperature [K]
    snowfall : float
        Fresh snowfall [mm]
    swe : float
        Current SWE [mm]
    dt : float
        Timestep [seconds]
    grain_radius : float
        Current grain radius [μm]
    r_min, r_max : float
        Min/max grain radius [μm]
    dr_dry, dr_wet : float
        Grain growth rates [μm/day]
    albedo_ref : float
        Albedo at reference grain size (100 μm)
    """
    if grain_radius is None:
        grain_radius = r_min

    dt_days = dt / 86400.0

    # Grain growth
    if t_snow >= TFRZ - 0.5:
        # Wet/melting metamorphism - fast growth
        grain_radius = grain_radius + dr_wet * dt_days
    else:
        # Dry metamorphism - slow growth
        grain_radius = grain_radius + dr_dry * dt_days

    # Fresh snow resets grain size (weighted)
    if snowfall > 0 and swe > 0:
        frac_new = snowfall / (swe + snowfall)
        grain_radius = frac_new * r_min + (1 - frac_new) * grain_radius

    # Limit grain size
    grain_radius = np.clip(grain_radius, r_min, r_max)

    # Albedo decreases ~0.1 per doubling of grain radius (from 100 μm ref)
    albedo = albedo_ref - 0.12 * np.log2(grain_radius / 100.0)
    albedo = np.clip(albedo, 0.45, 0.95)

    return albedo, grain_radius


def thin_snow_correction(albedo_snow, depth, albedo_ground=0.25, z_crit=0.02):
    """
    Blend snow and ground albedo for thin snowpacks.

    Uses snow optical depth concept - snow becomes optically thick
    at roughly 2-3 cm depth.

    Parameters
    ----------
    albedo_snow : float
        Snow albedo
    depth : float
        Snow depth [m]
    albedo_ground : float
        Bare ground albedo
    z_crit : float
        Critical depth for full snow albedo [m]
    """
    if depth >= z_crit:
        return albedo_snow

    if depth <= 0:
        return albedo_ground

    # Linear blending based on fraction of critical depth
    # Snow becomes optically thick quickly
    frac_snow = depth / z_crit
    return frac_snow * albedo_snow + (1 - frac_snow) * albedo_ground
