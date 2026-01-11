"""
Rain-snow partitioning schemes.

Available methods:
- 'threshold': Simple temperature threshold (CLM-style)
- 'linear': Linear ramp between thresholds (CLM Round 2)
- 'jennings': Humidity-dependent logistic (Jennings et al. 2018)
- 'wet_bulb': Wet-bulb temperature threshold (CoLM option)
"""

import numpy as np
from .constants import TFRZ, R_AIR, CP_AIR, LV_VAPORIZATION


def partition_rain_snow(precip, t_air, rh=None, pressure=None, method='linear', **kwargs):
    """
    Partition precipitation into rain and snow.

    Parameters
    ----------
    precip : float
        Total precipitation rate [mm/hr or kg/m²/s]
    t_air : float
        Air temperature [K]
    rh : float, optional
        Relative humidity [0-1], required for 'jennings' and 'wet_bulb'
    pressure : float, optional
        Air pressure [Pa], required for 'wet_bulb'
    method : str
        Partitioning method
    **kwargs : dict
        Method-specific parameters

    Returns
    -------
    snow_frac : float
        Fraction of precipitation falling as snow [0-1]
    """

    if method == 'threshold':
        return _threshold(t_air, **kwargs)
    elif method == 'linear':
        return _linear_ramp(t_air, **kwargs)
    elif method == 'jennings':
        if rh is None:
            raise ValueError("RH required for Jennings method")
        return _jennings(t_air, rh, **kwargs)
    elif method == 'wet_bulb':
        if rh is None or pressure is None:
            raise ValueError("RH and pressure required for wet_bulb method")
        return _wet_bulb(t_air, rh, pressure, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _threshold(t_air, t_snow=TFRZ, **kwargs):
    """
    Simple temperature threshold.
    All snow below t_snow, all rain above.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    t_snow : float
        Temperature threshold for 100% snow [K]
    """
    return 1.0 if t_air <= t_snow else 0.0


def _linear_ramp(t_air, t_all_snow=None, t_all_rain=None, **kwargs):
    """
    Linear ramp between temperature thresholds (CLM-style).

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    t_all_snow : float
        Temperature for 100% snow [K], default TFRZ + 1
    t_all_rain : float
        Temperature for 100% rain [K], default TFRZ + 3
    """
    if t_all_snow is None:
        t_all_snow = TFRZ + 1.0
    if t_all_rain is None:
        t_all_rain = TFRZ + 3.0

    if t_air <= t_all_snow:
        return 1.0
    elif t_air >= t_all_rain:
        return 0.0
    else:
        return (t_all_rain - t_air) / (t_all_rain - t_all_snow)


def _jennings(t_air, rh, a=-10.04, b=1.41, g=0.09, **kwargs):
    """
    Jennings et al. (2018) humidity-dependent logistic model.

    P(snow) = 1 / (1 + exp(a + b*T + g*RH))

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    rh : float
        Relative humidity [0-1]
    a, b, g : float
        Logistic regression coefficients
    """
    t_celsius = t_air - TFRZ
    rh_pct = rh * 100.0

    prob_snow = 1.0 / (1.0 + np.exp(a + b * t_celsius + g * rh_pct))

    # Binary classification
    return 1.0 if prob_snow >= 0.5 else 0.0


def _jennings_continuous(t_air, rh, a=-10.04, b=1.41, g=0.09, **kwargs):
    """
    Jennings model returning continuous probability (not binary).
    Useful for partial snow/rain mixing.
    """
    t_celsius = t_air - TFRZ
    rh_pct = rh * 100.0

    return 1.0 / (1.0 + np.exp(a + b * t_celsius + g * rh_pct))


def _wet_bulb(t_air, rh, pressure, t_threshold=TFRZ, **kwargs):
    """
    Wet-bulb temperature threshold (CoLM option).

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    rh : float
        Relative humidity [0-1]
    pressure : float
        Air pressure [Pa]
    t_threshold : float
        Wet-bulb threshold for snow [K]
    """
    t_wb = _calc_wet_bulb(t_air, rh, pressure)
    return 1.0 if t_wb <= t_threshold else 0.0


def _calc_wet_bulb(t_air, rh, pressure):
    """
    Calculate wet-bulb temperature using psychrometric approximation.
    """
    t_c = t_air - TFRZ

    # Saturation vapor pressure [Pa] (Tetens formula)
    if t_c >= 0:
        es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    else:
        es = 611.2 * np.exp(21.8745584 * t_c / (t_c + 265.5))

    # Actual vapor pressure
    e = rh * es

    # Psychrometric constant
    gamma = CP_AIR * pressure / (0.622 * LV_VAPORIZATION)

    # Wet-bulb approximation (Stull 2011)
    t_wb_c = t_c * np.arctan(0.151977 * np.sqrt(rh * 100 + 8.313659)) \
             + np.arctan(t_c + rh * 100) - np.arctan(rh * 100 - 1.676331) \
             + 0.00391838 * (rh * 100) ** 1.5 * np.arctan(0.023101 * rh * 100) \
             - 4.686035

    return t_wb_c + TFRZ


def calc_rh_from_specific_humidity(q, t_air, pressure):
    """
    Calculate relative humidity from specific humidity.

    Parameters
    ----------
    q : float
        Specific humidity [kg/kg]
    t_air : float
        Air temperature [K]
    pressure : float
        Air pressure [Pa]

    Returns
    -------
    rh : float
        Relative humidity [0-1]
    """
    t_c = t_air - TFRZ

    # Saturation vapor pressure [Pa]
    if t_c >= 0:
        es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    else:
        es = 611.2 * np.exp(21.8745584 * t_c / (t_c + 265.5))

    # Saturation specific humidity
    qs = 0.622 * es / (pressure - 0.378 * es)

    # Relative humidity
    rh = np.clip(q / max(qs, 1e-10), 0.0, 1.0)

    return rh
