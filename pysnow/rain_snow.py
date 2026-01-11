"""
Rain-snow partitioning schemes.

Available methods:
- 'threshold': Simple temperature threshold (CLM-style)
- 'linear': Linear ramp between thresholds (CLM Round 2)
- 'jennings': Humidity-dependent logistic (Jennings et al. 2018)
- 'dai': Hyperbolic tangent (Dai 2008) - global precipitation study
- 'wetbulb': Wet-bulb temperature threshold (Wang et al. 2019) - recommended for dry mountains
- 'wetbulb_linear': Wet-bulb with linear transition zone
- 'wetbulb_sigmoid': Wet-bulb sigmoid (Wang et al. 2019) - continuous probability

References:
- Dai (2008) J. Climate, doi:10.1175/2008JCLI2299.1 - Global precipitation analysis
- Jennings et al. (2018) Nature Communications, doi:10.1038/s41467-018-03629-7
- Wang et al. (2019) GRL, doi:10.1029/2019GL085722 - Wet-bulb method
- Stull (2011) J. Appl. Meteor. Climatol. - Wet-bulb temperature formula
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
    elif method == 'dai':
        return _dai(t_air, **kwargs)
    elif method in ('wet_bulb', 'wetbulb'):
        if rh is None or pressure is None:
            raise ValueError("RH and pressure required for wetbulb method")
        return _wetbulb(t_air, rh, pressure, **kwargs)
    elif method == 'wetbulb_linear':
        if rh is None or pressure is None:
            raise ValueError("RH and pressure required for wetbulb_linear method")
        return _wetbulb_linear(t_air, rh, pressure, **kwargs)
    elif method == 'wetbulb_sigmoid':
        if rh is None or pressure is None:
            raise ValueError("RH and pressure required for wetbulb_sigmoid method")
        return _wetbulb_sigmoid(t_air, rh, pressure, **kwargs)
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


def _dai(t_air, a=48.2292, b=0.7205, c=1.1662, **kwargs):
    """
    Dai (2008) hyperbolic tangent rain-snow partitioning.

    Uses a smooth hyperbolic tangent transition based on global precipitation
    analysis. The 50% rain/snow threshold is approximately 1.17°C.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    a : float
        Amplitude coefficient (controls steepness), default 48.2292
    b : float
        Slope coefficient [K^-1], default 0.7205
    c : float
        Temperature offset (50% threshold) [°C], default 1.1662

    Returns
    -------
    float
        Snow fraction [0-1]

    Reference
    ---------
    Dai (2008) J. Climate, doi:10.1175/2008JCLI2299.1
    """
    t_celsius = t_air - TFRZ

    # Hyperbolic tangent form: snow fraction decreases as T increases
    # f_s = 0.5 * (1 - tanh(b*(T-c))) gives proper asymptotic behavior
    # Scaled version to match Dai's coefficients
    snow_frac = 0.5 - (a * np.tanh(b * (t_celsius - c))) / 100.0

    return np.clip(snow_frac, 0.0, 1.0)


def _wetbulb(t_air, rh, pressure, tw_threshold=None, **kwargs):
    """
    Wet-bulb temperature threshold (Wang et al. 2019).

    Uses wet-bulb temperature instead of air temperature for rain-snow
    partitioning. This is more accurate in dry environments because
    falling hydrometeors cool via evaporation, so their surface temperature
    is closer to Tw than Ta.

    In dry air (low RH), Tw is significantly depressed below Ta, meaning
    snow can persist even when Ta > 0°C.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    rh : float
        Relative humidity [0-1]
    pressure : float
        Air pressure [Pa]
    tw_threshold : float, optional
        Wet-bulb threshold for snow [K]. Default is TFRZ + 1.0 (1°C)
        per Wang et al. (2019) recommendation.

    Returns
    -------
    float
        Snow fraction (0 or 1)

    Reference
    ---------
    Wang et al. (2019) GRL, doi:10.1029/2019GL085722
    """
    if tw_threshold is None:
        tw_threshold = TFRZ + 1.0  # Wang et al. recommend ~1°C

    t_wb = _calc_wet_bulb(t_air, rh, pressure)
    return 1.0 if t_wb <= tw_threshold else 0.0


def _wetbulb_linear(t_air, rh, pressure, tw_all_snow=None, tw_all_rain=None, **kwargs):
    """
    Wet-bulb temperature with linear transition zone.

    Combines wet-bulb approach with smooth transition rather than
    binary threshold. May be more physically realistic for mixed-phase
    precipitation events.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    rh : float
        Relative humidity [0-1]
    pressure : float
        Air pressure [Pa]
    tw_all_snow : float, optional
        Wet-bulb temp below which all precip is snow [K]. Default TFRZ (0°C)
    tw_all_rain : float, optional
        Wet-bulb temp above which all precip is rain [K]. Default TFRZ + 2 (2°C)

    Returns
    -------
    float
        Snow fraction [0-1]
    """
    if tw_all_snow is None:
        tw_all_snow = TFRZ
    if tw_all_rain is None:
        tw_all_rain = TFRZ + 2.0

    t_wb = _calc_wet_bulb(t_air, rh, pressure)

    if t_wb <= tw_all_snow:
        return 1.0
    elif t_wb >= tw_all_rain:
        return 0.0
    else:
        return (tw_all_rain - t_wb) / (tw_all_rain - tw_all_snow)


def _wetbulb_sigmoid(t_air, rh, pressure, a=6.99e-5, b=2.0, c=3.97, **kwargs):
    """
    Wang et al. (2019) wet-bulb sigmoid rain-snow partitioning.

    Uses a smooth sigmoid transition based on wet-bulb temperature.
    More physically realistic than a binary threshold, providing
    continuous probability of snow vs rain.

    The 50% threshold occurs at approximately Tw = -0.8°C.

    Parameters
    ----------
    t_air : float
        Air temperature [K]
    rh : float
        Relative humidity [0-1]
    pressure : float
        Air pressure [Pa]
    a : float
        Sigmoid coefficient, default 6.99e-5
    b : float
        Sigmoid exponent coefficient [K^-1], default 2.0
    c : float
        Temperature offset [K], default 3.97

    Returns
    -------
    float
        Snow fraction [0-1]

    Reference
    ---------
    Wang et al. (2019) GRL, doi:10.1029/2019GL085722
    """
    t_wb = _calc_wet_bulb(t_air, rh, pressure)
    t_wb_celsius = t_wb - TFRZ

    # Sigmoid formula: f_s = 1 / (1 + a * exp(b * (T_w + c)))
    # At low Tw: exp term → 0, f_s → 1 (all snow)
    # At high Tw: exp term → large, f_s → 0 (all rain)
    exp_term = a * np.exp(b * (t_wb_celsius + c))
    snow_frac = 1.0 / (1.0 + exp_term)

    return np.clip(snow_frac, 0.0, 1.0)


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
