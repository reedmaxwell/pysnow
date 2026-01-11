"""
Snow thermal properties and phase change.

Includes:
- Thermal conductivity
- Heat capacity
- Phase change (melt/freeze) with energy hierarchy
- Cold content tracking
"""

import numpy as np
from .constants import (TFRZ, RHO_WATER, RHO_ICE, CP_ICE, CP_WATER,
                        TK_AIR, TK_ICE, LF_FUSION)


# =============================================================================
# Thermal Conductivity
# =============================================================================

def calc_thermal_conductivity(density, method='jordan', **kwargs):
    """
    Calculate snow thermal conductivity.

    Parameters
    ----------
    density : float
        Snow density [kg/m³]
    method : str
        'jordan': Jordan (1991) - CLM/CoLM default
        'sturm': Sturm et al. (1997) - seasonal snow
        'calonne': Calonne et al. (2011) - microstructure-based

    Returns
    -------
    thk : float
        Thermal conductivity [W/(m·K)]
    """
    if method == 'jordan':
        return _jordan_conductivity(density)
    elif method == 'sturm':
        return _sturm_conductivity(density)
    elif method == 'calonne':
        return _calonne_conductivity(density)
    else:
        raise ValueError(f"Unknown conductivity method: {method}")


def _jordan_conductivity(density):
    """
    Jordan (1991) thermal conductivity formula.
    Used by CLM and CoLM.

    Parameters
    ----------
    density : float
        Snow density [kg/m³]
    """
    bw = density  # bulk density = partial density for dry snow
    thk = TK_AIR + (7.75e-5 * bw + 1.105e-6 * bw**2) * (TK_ICE - TK_AIR)
    return thk


def _sturm_conductivity(density):
    """
    Sturm et al. (1997) for seasonal snow.
    """
    rho = density
    if rho < 156:
        thk = 0.023 + 0.234 * (rho / 1000)
    else:
        thk = 0.138 - 1.01 * (rho / 1000) + 3.233 * (rho / 1000)**2
    return thk


def _calonne_conductivity(density):
    """
    Calonne et al. (2011) empirical fit.
    """
    rho = density
    thk = 2.5e-6 * rho**2 - 1.23e-4 * rho + 0.024
    return max(0.025, thk)


# =============================================================================
# Heat Capacity
# =============================================================================

def calc_heat_capacity(swe, t_snow):
    """
    Calculate snow heat capacity.

    Parameters
    ----------
    swe : float
        Snow water equivalent [mm or kg/m²]
    t_snow : float
        Snow temperature [K]

    Returns
    -------
    cv : float
        Heat capacity [J/(m²·K)]
    """
    # Assume all ice when T < TFRZ, mix when T = TFRZ
    if t_snow < TFRZ:
        cp = CP_ICE
    else:
        cp = CP_ICE  # At 0°C, still mostly ice

    # swe in mm = kg/m², cp in J/(kg·K)
    cv = swe * cp

    return cv


# =============================================================================
# Cold Content
# =============================================================================

def calc_cold_content(swe, t_snow):
    """
    Calculate snow cold content (energy deficit from 0°C).

    Negative values indicate energy needed to warm snow to melting point.

    Parameters
    ----------
    swe : float
        Snow water equivalent [mm or kg/m²]
    t_snow : float
        Snow temperature [K]

    Returns
    -------
    cold_content : float
        Cold content [J/m²], negative when T < 0°C
    """
    t_deficit = t_snow - TFRZ  # Negative when cold
    cold_content = swe * CP_ICE * t_deficit

    return cold_content


def temp_from_cold_content(cold_content, swe):
    """
    Calculate snow temperature from cold content.

    Parameters
    ----------
    cold_content : float
        Cold content [J/m²]
    swe : float
        Snow water equivalent [mm or kg/m²]

    Returns
    -------
    t_snow : float
        Snow temperature [K]
    """
    if swe <= 0:
        return TFRZ

    t_deficit = cold_content / (swe * CP_ICE)
    t_snow = TFRZ + t_deficit

    # Limit temperature to physical range
    t_snow = max(173.15, min(TFRZ, t_snow))  # -100°C to 0°C
    return t_snow


# =============================================================================
# Phase Change - Energy Hierarchy (SNOWCLIM-style)
# =============================================================================

def apply_energy_to_snowpack(energy, swe, t_snow, liquid_water,
                             method='hierarchy', **kwargs):
    """
    Apply energy to snowpack with proper phase change accounting.

    Parameters
    ----------
    energy : float
        Net energy flux [J/m²], positive = warming
    swe : float
        Snow water equivalent [mm]
    t_snow : float
        Snow temperature [K]
    liquid_water : float
        Liquid water in snowpack [mm]
    method : str
        'hierarchy': SNOWCLIM-style (CC → refreeze → melt)
        'clm': CLM-style (simultaneous)

    Returns
    -------
    dict with:
        swe : float - Updated SWE [mm]
        t_snow : float - Updated temperature [K]
        liquid_water : float - Updated liquid water [mm]
        melt : float - Melt amount [mm]
        refreeze : float - Refreeze amount [mm]
    """
    if method == 'hierarchy':
        return _energy_hierarchy(energy, swe, t_snow, liquid_water, **kwargs)
    elif method == 'clm':
        return _energy_clm_style(energy, swe, t_snow, liquid_water, **kwargs)
    else:
        raise ValueError(f"Unknown phase change method: {method}")


def _energy_hierarchy(energy, swe, t_snow, liquid_water,
                      cold_content_tax=False, min_swe_thermal=5.0, **kwargs):
    """
    SNOWCLIM-style energy hierarchy:
    1. Cold content warming
    2. Refreezing of liquid water
    3. Melting

    Parameters
    ----------
    energy : float
        Net energy [J/m²]
    swe : float
        Snow water equivalent [mm]
    t_snow : float
        Snow temperature [K]
    liquid_water : float
        Liquid water [mm]
    cold_content_tax : bool
        Apply SNOWCLIM cold content tax (energy conservation)
    min_swe_thermal : float
        Minimum effective SWE for thermal calculations [mm]
        Prevents numerical instability with thin snow
    """
    melt = 0.0
    refreeze = 0.0
    ice = swe - liquid_water  # Ice portion

    # Use minimum effective mass for thermal stability
    # This prevents wild temperature swings with thin snow
    ice_effective = max(ice, min_swe_thermal) if ice > 0 else ice

    # Current cold content (use effective mass for temperature calc)
    cc = calc_cold_content(ice_effective, t_snow)

    # Optional: Cold content tax to prevent unrealistic cold packs
    if cold_content_tax and energy < 0 and cc < -10000:
        # Reduce energy applied to cold content
        tax_rate = min(0.9, abs(cc) / 100000)
        energy = energy * (1 - tax_rate)

    # Step 1: Cold content warming (if energy > 0 and snow is cold)
    if energy > 0 and t_snow < TFRZ and ice > 0:
        # Energy needed to warm to 0°C
        energy_to_zero = -cc  # cc is negative, so this is positive

        if energy >= energy_to_zero:
            # Warm to 0°C, excess goes to melt
            energy = energy - energy_to_zero
            t_snow = TFRZ
            cc = 0
        else:
            # Partial warming
            cc = cc + energy
            t_snow = temp_from_cold_content(cc, ice_effective)
            energy = 0

    # Step 2: Refreezing (if energy < 0 and liquid water exists)
    if energy < 0 and liquid_water > 0 and t_snow <= TFRZ:
        # Energy that can be released by refreezing
        max_refreeze_energy = liquid_water * RHO_WATER * LF_FUSION / 1000  # Convert mm to m

        if abs(energy) >= max_refreeze_energy:
            # Refreeze all liquid water
            refreeze = liquid_water
            energy = energy + max_refreeze_energy
            liquid_water = 0
            # Remaining negative energy cools the snow
            if ice + refreeze > 0:
                cc = cc + energy
                ice_eff_new = max(ice + refreeze, min_swe_thermal)
                t_snow = temp_from_cold_content(cc, ice_eff_new)
            energy = 0
        else:
            # Partial refreeze
            refreeze = abs(energy) / (RHO_WATER * LF_FUSION / 1000)
            liquid_water = liquid_water - refreeze
            energy = 0
            t_snow = TFRZ

    # Step 2b: Cooling (if energy < 0 and no liquid water)
    if energy < 0 and liquid_water <= 0 and ice > 0:
        cc = cc + energy
        t_snow = temp_from_cold_content(cc, ice_effective)
        energy = 0

    # Step 3: Melting (if energy > 0 and at melting point)
    if energy > 0 and t_snow >= TFRZ and ice > 0:
        # Maximum possible melt
        max_melt = ice
        melt_energy = max_melt * RHO_WATER * LF_FUSION / 1000

        if energy >= melt_energy:
            # Melt all ice
            melt = max_melt
            energy = energy - melt_energy
        else:
            # Partial melt
            melt = energy / (RHO_WATER * LF_FUSION / 1000)
            energy = 0

        # Melt becomes liquid water
        liquid_water = liquid_water + melt

    # Update SWE (ice + liquid)
    ice = ice - melt + refreeze
    swe = ice + liquid_water

    return {
        'swe': max(0, swe),
        't_snow': t_snow,
        'liquid_water': max(0, liquid_water),
        'melt': melt,
        'refreeze': refreeze,
        'ice': max(0, ice)
    }


def _energy_clm_style(energy, swe, t_snow, liquid_water, min_swe_thermal=5.0, **kwargs):
    """
    CLM-style phase change (simplified).
    """
    melt = 0.0
    refreeze = 0.0
    ice = swe - liquid_water

    # Use minimum effective mass for thermal stability
    ice_effective = max(ice, min_swe_thermal) if ice > 0 else ice

    # Warming or cooling
    if energy > 0:
        # Warming
        if t_snow < TFRZ:
            # First warm to 0°C
            cc = calc_cold_content(ice_effective, t_snow)
            energy_to_zero = -cc

            if energy >= energy_to_zero:
                t_snow = TFRZ
                energy = energy - energy_to_zero
            else:
                t_snow = temp_from_cold_content(cc + energy, ice_effective)
                energy = 0

        # Then melt
        if energy > 0 and ice > 0 and t_snow >= TFRZ:
            melt = min(ice, energy / (RHO_WATER * LF_FUSION / 1000))
            liquid_water = liquid_water + melt
            ice = ice - melt

    else:
        # Cooling - refreeze then cool
        if liquid_water > 0:
            max_refreeze = abs(energy) / (RHO_WATER * LF_FUSION / 1000)
            refreeze = min(liquid_water, max_refreeze)
            liquid_water = liquid_water - refreeze
            ice = ice + refreeze
            energy = energy + refreeze * RHO_WATER * LF_FUSION / 1000

        if energy < 0 and ice > 0:
            ice_eff_new = max(ice, min_swe_thermal)
            cc = calc_cold_content(ice_eff_new, t_snow)
            t_snow = temp_from_cold_content(cc + energy, ice_eff_new)

    swe = ice + liquid_water

    return {
        'swe': max(0, swe),
        't_snow': min(TFRZ, t_snow),
        'liquid_water': max(0, liquid_water),
        'melt': melt,
        'refreeze': refreeze,
        'ice': max(0, ice)
    }


# =============================================================================
# Liquid Water Management
# =============================================================================

def update_liquid_water(liquid_water, depth, dt,
                        min_frac=0.01, max_frac=0.10,
                        drain_rate=2.778e-5, **kwargs):
    """
    Update liquid water in snowpack with drainage.

    Parameters
    ----------
    liquid_water : float
        Current liquid water [mm]
    depth : float
        Snow depth [m]
    dt : float
        Timestep [s]
    min_frac : float
        Irreducible water fraction (of depth)
    max_frac : float
        Maximum water fraction
    drain_rate : float
        Drainage rate [m/s], default 10 cm/hr

    Returns
    -------
    liquid_water : float
        Updated liquid water [mm]
    runoff : float
        Drainage/runoff [mm]
    """
    if depth <= 0:
        return 0, liquid_water

    # Convert depth to mm for comparison
    depth_mm = depth * 1000

    min_water = min_frac * depth_mm
    max_water = max_frac * depth_mm

    runoff = 0

    # Drain excess above max immediately
    if liquid_water > max_water:
        runoff = liquid_water - max_water
        liquid_water = max_water

    # Gradual drainage between min and max
    elif liquid_water > min_water:
        drain = drain_rate * 1000 * dt  # Convert to mm
        drain = min(drain, liquid_water - min_water)
        liquid_water = liquid_water - drain
        runoff = drain

    return max(0, liquid_water), runoff
