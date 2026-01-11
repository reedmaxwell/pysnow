"""
Radiation balance calculations.

Includes shortwave and longwave radiation.
"""

import numpy as np
from .constants import STEFAN_BOLTZMANN, EMISSIVITY_SNOW, TFRZ


def calc_net_radiation(sw_down, lw_down, albedo, t_snow,
                       emissivity=EMISSIVITY_SNOW, **kwargs):
    """
    Calculate net radiation at snow surface.

    Parameters
    ----------
    sw_down : float
        Incoming shortwave radiation [W/m²]
    lw_down : float
        Incoming longwave radiation [W/m²]
    albedo : float
        Snow albedo [0-1]
    t_snow : float
        Snow surface temperature [K]
    emissivity : float
        Snow emissivity

    Returns
    -------
    dict with:
        Rn : float - Net radiation [W/m²]
        sw_net : float - Net shortwave [W/m²]
        lw_net : float - Net longwave [W/m²]
        lw_up : float - Outgoing longwave [W/m²]
    """
    # Net shortwave (absorbed)
    sw_net = sw_down * (1.0 - albedo)

    # Outgoing longwave
    lw_up = emissivity * STEFAN_BOLTZMANN * t_snow ** 4

    # Net longwave (incoming - outgoing + reflected)
    # Some LW is reflected: (1 - emissivity) * lw_down
    lw_net = emissivity * lw_down - lw_up

    # Total net radiation
    Rn = sw_net + lw_net

    return {
        'Rn': Rn,
        'sw_net': sw_net,
        'lw_net': lw_net,
        'sw_up': sw_down * albedo,
        'lw_up': lw_up
    }


def calc_surface_temperature(t_snow, energy_balance, heat_capacity, dt,
                             t_max=TFRZ, **kwargs):
    """
    Update surface temperature based on energy balance.

    For snow, temperature is capped at freezing point.

    Parameters
    ----------
    t_snow : float
        Current temperature [K]
    energy_balance : float
        Net energy [W/m²]
    heat_capacity : float
        Heat capacity [J/(m²·K)]
    dt : float
        Timestep [s]
    t_max : float
        Maximum temperature [K]

    Returns
    -------
    t_new : float
        Updated temperature [K]
    excess_energy : float
        Energy not used for temperature change [J/m²]
    """
    if heat_capacity <= 0:
        return t_snow, energy_balance * dt

    # Temperature change
    dt_temp = energy_balance * dt / heat_capacity
    t_new = t_snow + dt_temp

    excess_energy = 0.0

    # Cap at melting point
    if t_new > t_max:
        excess_energy = (t_new - t_max) * heat_capacity
        t_new = t_max

    return t_new, excess_energy
