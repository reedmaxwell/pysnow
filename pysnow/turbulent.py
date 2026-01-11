"""
Turbulent heat flux calculations.

Includes sensible and latent heat fluxes using:
- Bulk aerodynamic formulation
- Richardson number stability correction
- Optional windless exchange coefficient (SNOWCLIM)
"""

import numpy as np
from .constants import (TFRZ, CP_AIR, RHO_AIR_SL, VON_KARMAN, GRAVITY,
                        R_AIR, LV_VAPORIZATION, LS_SUBLIMATION)


def calc_turbulent_fluxes(t_air, t_snow, wind, rh, pressure,
                          z_wind=10.0, z_temp=2.0, z_0=0.001, z_h=None,
                          stability_correction=True, windless_coeff=0.0,
                          **kwargs):
    """
    Calculate sensible and latent heat fluxes.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    t_snow : float
        Snow surface temperature [K]
    wind : float
        Wind speed [m/s]
    rh : float
        Relative humidity [0-1]
    pressure : float
        Air pressure [Pa]
    z_wind : float
        Wind measurement height [m]
    z_temp : float
        Temperature measurement height [m]
    z_0 : float
        Momentum roughness length [m]
    z_h : float
        Heat/vapor roughness length [m], default z_0/10
    stability_correction : bool
        Apply Richardson number correction
    windless_coeff : float
        Windless exchange coefficient [W/(m²·K)], SNOWCLIM E0

    Returns
    -------
    dict with:
        H : float - Sensible heat flux [W/m²], positive = toward surface
        LE : float - Latent heat flux [W/m²], positive = toward surface
        E : float - Sublimation/evaporation rate [mm/hr]
    """
    if z_h is None:
        z_h = z_0 / 10.0

    # Air density
    rho_air = pressure / (R_AIR * t_air)

    # Neutral exchange coefficient
    Ch_n = (VON_KARMAN ** 2) / (np.log(z_wind / z_0) * np.log(z_temp / z_h))

    # Stability correction using Richardson number
    if stability_correction and wind > 0.1:
        Rib = _bulk_richardson(t_air, t_snow, wind, z_wind)
        fh = _stability_function(Rib, Ch_n, z_wind, z_0)
    else:
        fh = 1.0

    Ch = Ch_n * fh

    # Effective exchange (with windless coefficient)
    wind_eff = max(wind, 0.1)  # Minimum wind
    exchange = rho_air * Ch * wind_eff

    if windless_coeff > 0:
        # Add windless exchange (convert from W/(m²·K) to same units)
        exchange = exchange + windless_coeff / CP_AIR

    # Sensible heat flux [W/m²]
    # Positive when air warmer than snow (heating snow)
    H = CP_AIR * exchange * (t_air - t_snow)

    # Latent heat flux
    # Vapor pressure deficit
    e_air = rh * _sat_vapor_pressure(t_air)
    e_snow = _sat_vapor_pressure(t_snow)  # Snow surface saturated

    # Specific humidity
    q_air = 0.622 * e_air / (pressure - 0.378 * e_air)
    q_snow = 0.622 * e_snow / (pressure - 0.378 * e_snow)

    # Mass flux [kg/(m²·s)]
    E_mass = exchange * (q_air - q_snow)

    # Latent heat depends on snow temperature
    if t_snow < TFRZ:
        L = LS_SUBLIMATION  # Sublimation
    else:
        L = LV_VAPORIZATION  # Evaporation

    # Latent heat flux [W/m²]
    LE = L * E_mass

    # Sublimation rate [mm/hr] (negative = mass loss)
    E_rate = E_mass * 3600.0  # Convert to mm/hr

    return {
        'H': H,
        'LE': LE,
        'E': E_rate,
        'E_mass': E_mass
    }


def _bulk_richardson(t_air, t_snow, wind, z):
    """
    Calculate bulk Richardson number.

    Rib > 0: Stable (snow colder than air)
    Rib < 0: Unstable (snow warmer than air)
    """
    dt = t_air - t_snow
    t_mean = (t_air + t_snow) / 2.0

    Rib = GRAVITY * z * dt / (t_mean * wind ** 2)

    # Limit extreme values
    return np.clip(Rib, -10.0, 1.0)


def _stability_function(Rib, Ch_n, z, z_0, c=5.0):
    """
    Louis (1979) stability correction.

    Parameters
    ----------
    Rib : float
        Bulk Richardson number
    Ch_n : float
        Neutral exchange coefficient
    z : float
        Reference height [m]
    z_0 : float
        Roughness length [m]
    c : float
        Louis parameter (typically 5)
    """
    if Rib < 0:
        # Unstable
        fh = 1.0 - (3.0 * c * Rib) / (1.0 + 3.0 * c**2 * Ch_n * np.sqrt(-Rib * z / z_0))
    elif Rib > 0:
        # Stable
        fh = 1.0 / (1.0 + 2.0 * c * Rib / np.sqrt(1.0 + Rib))
    else:
        # Neutral
        fh = 1.0

    # Limit
    return np.clip(fh, 0.01, 10.0)


def _sat_vapor_pressure(T):
    """
    Saturation vapor pressure [Pa] using Tetens formula.

    Parameters
    ----------
    T : float
        Temperature [K]
    """
    t_c = T - TFRZ

    if t_c >= 0:
        # Over water
        es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    else:
        # Over ice
        es = 611.2 * np.exp(21.8745584 * t_c / (t_c + 265.5))

    return es


def calc_ground_heat_flux(t_snow, t_soil, depth, thk_snow, thk_soil=1.0,
                          method='gradient', G_const=2.0, **kwargs):
    """
    Calculate ground heat flux at snow-soil interface.

    Parameters
    ----------
    t_snow : float
        Snow base temperature [K]
    t_soil : float
        Soil surface temperature [K]
    depth : float
        Snow depth [m]
    thk_snow : float
        Snow thermal conductivity [W/(m·K)]
    thk_soil : float
        Soil thermal conductivity [W/(m·K)]
    method : str
        'gradient': Temperature gradient method
        'constant': Fixed value (SNOWCLIM default)
    G_const : float
        Constant ground heat flux [W/m²]

    Returns
    -------
    G : float
        Ground heat flux [W/m²], positive = toward snow
    """
    if method == 'constant':
        return G_const

    elif method == 'gradient':
        if depth <= 0:
            return 0.0

        # Interface conductivity (harmonic mean)
        dz_snow = depth / 2.0
        dz_soil = 0.1  # Assume top 10 cm of soil

        thk_interface = (dz_snow + dz_soil) / (dz_snow / thk_snow + dz_soil / thk_soil)

        # Heat flux
        G = thk_interface * (t_soil - t_snow) / (dz_snow + dz_soil)

        return G

    else:
        return G_const
