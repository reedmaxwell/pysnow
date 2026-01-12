#!/usr/bin/env python3
"""
Comprehensive SNOTEL Model Evaluation Framework

Inspired by Ryken et al. (2020) Advances in Water Resources:
"Sensitivity and model reduction of simulated snow processes"

Tests PySnow against multiple SNOTEL sites across:
- Different decades (1990s, 2000s, 2010s, 2020s)
- Different regions and climates
- Multiple parameterization combinations

Computes comprehensive metrics:
- Peak SWE magnitude and timing
- Melt-out (snow-free) date
- RMSE, NSE, correlation
- Early season accumulation
- Forcing bias assessment (CW3E vs SNOTEL)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Plotting configs - fixed configs to always plot
FIXED_PLOT_CONFIGS = ['Wetbulb-Tw2C', 'Wetbulb-Damp0.7', 'CLM-baseline']
PLOT_COLORS = {
    'Wetbulb-Tw2C': 'tab:blue',
    'Wetbulb-Damp0.7': 'tab:green',
    'CLM-baseline': 'tab:red',
    'dynamic1': 'tab:purple',  # Best dynamic performer
    'dynamic2': 'tab:orange',  # 2nd best dynamic performer
}

# Add pysnow repo to path
pysnow_repo = Path("/Users/reed/Projects/pysnow_repo")
sys.path.insert(0, str(pysnow_repo))

from pysnow import PySnowModel, SnowConfig

# HydroData imports
try:
    import hf_hydrodata as hf
    import subsettools as st
    email = os.environ.get('HF_EMAIL', 'reedmm@princeton.edu')
    pin = os.environ.get('HF_PIN', '4321')
    hf.register_api_pin(email, pin)
    HAS_HYDRODATA = True
except ImportError:
    HAS_HYDRODATA = False
    print("Warning: hf_hydrodata/subsettools not available")


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class ForcingBias:
    """Forcing data bias assessment."""
    site_name: str
    water_year: int

    # Temperature bias (CW3E - SNOTEL)
    temp_bias_mean: float = np.nan  # Mean bias (K)
    temp_bias_std: float = np.nan   # Std dev of bias
    temp_rmse: float = np.nan
    temp_corr: float = np.nan

    # Precipitation assessment
    cw3e_total_precip: float = np.nan  # Total CW3E precip (mm)
    snotel_peak_swe: float = np.nan    # SNOTEL peak SWE as proxy
    precip_ratio: float = np.nan       # CW3E / SNOTEL peak (rough proxy)


@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics."""
    site_name: str
    site_id: str
    water_year: int
    config_name: str

    # Peak SWE metrics
    obs_peak_swe: float = np.nan
    mod_peak_swe: float = np.nan
    peak_swe_bias: float = np.nan       # mm
    peak_swe_bias_pct: float = np.nan   # %

    # Peak timing
    obs_peak_date: Optional[datetime] = None
    mod_peak_date: Optional[datetime] = None
    peak_date_bias_days: int = 0

    # Melt-out (snow-free) date
    obs_meltout_date: Optional[datetime] = None
    mod_meltout_date: Optional[datetime] = None
    meltout_bias_days: Optional[int] = None

    # Statistical metrics
    rmse: float = np.nan
    mae: float = np.nan
    correlation: float = np.nan
    nse: float = np.nan  # Nash-Sutcliffe Efficiency
    kge: float = np.nan  # Kling-Gupta Efficiency

    # Early season (Oct-Dec)
    early_season_obs_accum: float = np.nan
    early_season_mod_accum: float = np.nan
    early_season_bias_pct: float = np.nan

    # Late season (Apr-Jun)
    late_season_rmse: float = np.nan

    # Snow partitioning
    total_precip: float = np.nan
    total_snowfall: float = np.nan
    total_rainfall: float = np.nan
    snow_fraction: float = np.nan


# =============================================================================
# SITE AND YEAR DEFINITIONS
# =============================================================================

def get_snotel_sites() -> List[Dict]:
    """Return list of test sites spanning different regions."""
    return [
        # Sierra Nevada
        {"name": "CSS Lab", "id": "428:CA:SNTL", "state": "CA", "elev_m": 2100, "region": "Sierra",
         "lat": 39.31, "lon": -120.37},
        {"name": "Leavitt Lake", "id": "518:CA:SNTL", "state": "CA", "elev_m": 2987, "region": "Sierra",
         "lat": 38.28, "lon": -119.56},

        # Colorado Rockies
        {"name": "Berthoud Summit", "id": "335:CO:SNTL", "state": "CO", "elev_m": 3536, "region": "Rockies-CO",
         "lat": 39.80, "lon": -105.78},
        {"name": "Loveland Basin", "id": "602:CO:SNTL", "state": "CO", "elev_m": 3536, "region": "Rockies-CO",
         "lat": 39.68, "lon": -105.88},

        # Wyoming
        {"name": "Canyon", "id": "384:WY:SNTL", "state": "WY", "elev_m": 2438, "region": "Rockies-WY",
         "lat": 44.73, "lon": -110.50},
        {"name": "Togwotee Pass", "id": "822:WY:SNTL", "state": "WY", "elev_m": 2936, "region": "Rockies-WY",
         "lat": 43.76, "lon": -110.07},

        # Utah
        {"name": "Trial Lake", "id": "823:UT:SNTL", "state": "UT", "elev_m": 3018, "region": "Wasatch",
         "lat": 40.68, "lon": -110.95},

        # Pacific Northwest (maritime)
        {"name": "Paradise", "id": "679:WA:SNTL", "state": "WA", "elev_m": 1561, "region": "Cascades",
         "lat": 46.79, "lon": -121.74},
    ]


def get_water_years() -> List[int]:
    """Return water years spanning different decades."""
    return [
        1995,  # 1990s
        2005,  # 2000s
        2011,  # Early 2010s
        2017,  # Late 2010s
        2023,  # 2020s
        2024,  # Most recent
    ]


# =============================================================================
# MODEL CONFIGURATIONS - Comprehensive parameterization combinations
# =============================================================================

def get_model_configs() -> Dict[str, SnowConfig]:
    """
    Define model configurations testing different parameterization combinations.
    Inspired by Ryken et al. approach of perturbing multiple parameters.
    """
    configs = {}

    # ==========================================================================
    # BASELINE CONFIGURATIONS (no damping)
    # ==========================================================================

    # CLM baseline
    configs["CLM-baseline"] = SnowConfig(
        rain_snow_method='linear',
        rain_snow_params={'t_all_snow': 274.15, 't_all_rain': 276.15},
        albedo_method='clm_decay',
        phase_change_method='clm',
        thin_snow_damping=0.0,
        ground_flux_const=2.0,
    )

    # Wetbulb baseline
    configs["Wetbulb-baseline"] = SnowConfig(
        rain_snow_method='wetbulb',
        rain_snow_params={'tw_threshold': 274.15},
        albedo_method='snicar_lite',
        phase_change_method='hierarchy',
        thin_snow_damping=0.0,
        ground_flux_const=2.0,
    )

    # Jennings baseline
    configs["Jennings-baseline"] = SnowConfig(
        rain_snow_method='jennings',
        albedo_method='snicar_lite',
        phase_change_method='hierarchy',
        thin_snow_damping=0.0,
        ground_flux_const=2.0,
    )

    # ==========================================================================
    # DAMPING SENSITIVITY (with wetbulb)
    # ==========================================================================

    for damp in [0.3, 0.5, 0.7]:
        configs[f"Wetbulb-Damp{damp}"] = SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 274.15},
            albedo_method='snicar_lite',
            phase_change_method='hierarchy',
            thin_snow_damping=damp,
            thin_snow_threshold=75.0,
            ground_flux_const=2.0,
        )

    # ==========================================================================
    # DAMPING SENSITIVITY (with CLM linear)
    # ==========================================================================

    for damp in [0.3, 0.5, 0.7]:
        configs[f"CLM-Damp{damp}"] = SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 274.15, 't_all_rain': 276.15},
            albedo_method='clm_decay',
            phase_change_method='clm',
            thin_snow_damping=damp,
            thin_snow_threshold=75.0,
            ground_flux_const=2.0,
        )

    # ==========================================================================
    # ALBEDO METHOD SENSITIVITY (with wetbulb + damping)
    # ==========================================================================

    for albedo in ['clm_decay', 'vic', 'essery', 'snicar_lite']:
        configs[f"Wetbulb-{albedo}"] = SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 274.15},
            albedo_method=albedo,
            phase_change_method='hierarchy',
            thin_snow_damping=0.5,
            thin_snow_threshold=75.0,
            ground_flux_const=2.0,
        )

    # ==========================================================================
    # GROUND HEAT FLUX SENSITIVITY
    # ==========================================================================

    for ghf in [0.5, 1.0, 2.0, 3.0]:
        configs[f"Wetbulb-GHF{ghf}"] = SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 274.15},
            albedo_method='snicar_lite',
            phase_change_method='hierarchy',
            thin_snow_damping=0.5,
            thin_snow_threshold=75.0,
            ground_flux_const=ghf,
        )

    # ==========================================================================
    # RAIN-SNOW THRESHOLD SENSITIVITY (for wetbulb)
    # ==========================================================================

    for tw in [273.15, 274.15, 275.15]:  # 0C, 1C, 2C
        tw_name = f"{tw - 273.15:.0f}C"
        configs[f"Wetbulb-Tw{tw_name}"] = SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': tw},
            albedo_method='snicar_lite',
            phase_change_method='hierarchy',
            thin_snow_damping=0.5,
            thin_snow_threshold=75.0,
            ground_flux_const=2.0,
        )

    # ==========================================================================
    # OTHER RAIN-SNOW METHODS
    # ==========================================================================

    # Dai (2008) hyperbolic tangent
    configs["Dai-Damp0.5"] = SnowConfig(
        rain_snow_method='dai',
        albedo_method='snicar_lite',
        phase_change_method='hierarchy',
        thin_snow_damping=0.5,
        thin_snow_threshold=75.0,
        ground_flux_const=2.0,
    )

    # Wetbulb sigmoid (Wang et al. 2019)
    configs["WetbulbSig-Damp0.5"] = SnowConfig(
        rain_snow_method='wetbulb_sigmoid',
        albedo_method='snicar_lite',
        phase_change_method='hierarchy',
        thin_snow_damping=0.5,
        thin_snow_threshold=75.0,
        ground_flux_const=2.0,
    )

    return configs


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def get_forcing_data(site_id: str, wy: int) -> Optional[np.ndarray]:
    """Fetch CW3E forcing data for a site and water year."""
    if not HAS_HYDRODATA:
        return None

    try:
        # Get site location
        sites = hf.get_point_metadata(
            dataset="snotel", variable="swe",
            temporal_resolution="daily", aggregation="sod",
            site_ids=site_id
        )
        if sites.empty:
            return None

        lat = sites.iloc[0]['latitude']
        lon = sites.iloc[0]['longitude']

        # Date range
        start_date = f"{wy-1}-10-01"
        end_date = f"{wy}-09-30"

        # Get grid bounds
        latlon_bounds = [[lat, lon], [lat, lon]]
        bounds, _ = st.define_latlon_domain(latlon_bounds=latlon_bounds, grid="conus2")

        # Fetch forcing variables
        dataset = "CW3E"
        version = "1.0"

        def fetch_var(varname):
            options = {
                "dataset": dataset,
                "period": "hourly",
                "variable": varname,
                "start_time": start_date,
                "end_time": end_date,
                "grid_bounds": bounds,
                "dataset_version": version
            }
            return hf.get_gridded_data(options)

        DSWR = fetch_var("downward_shortwave")
        DLWR = fetch_var("downward_longwave")
        APCP = fetch_var("precipitation")
        Temp = fetch_var("air_temp")
        UGRD = fetch_var("east_windspeed")
        VGRD = fetch_var("north_windspeed")
        Press = fetch_var("atmospheric_pressure")
        SPFH = fetch_var("specific_humidity")

        n_hours = len(DSWR)
        APCP[APCP < 0] = 0.0

        forcing = np.zeros((n_hours, 8))
        forcing[:, 0] = DSWR[:, 0, 0]
        forcing[:, 1] = DLWR[:, 0, 0]
        forcing[:, 2] = APCP[:, 0, 0] * 3600  # mm/s -> mm/hr
        forcing[:, 3] = Temp[:, 0, 0]
        forcing[:, 4] = UGRD[:, 0, 0]
        forcing[:, 5] = VGRD[:, 0, 0]
        forcing[:, 6] = Press[:, 0, 0]
        forcing[:, 7] = SPFH[:, 0, 0]

        return forcing

    except Exception as e:
        print(f"    Error fetching forcing: {e}")
        return None


def get_snotel_obs(site_id: str, wy: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fetch SNOTEL observations: SWE and temperature.
    Returns: (dates, swe, temperature)
    """
    if not HAS_HYDRODATA:
        return None, None, None

    try:
        start_date = f"{wy-1}-10-01"
        end_date = f"{wy}-09-30"

        # Get SWE
        swe_data = hf.get_point_data(
            dataset="snotel",
            variable="swe",
            temporal_resolution="daily",
            aggregation="sod",
            site_ids=site_id,
            date_start=start_date,
            date_end=end_date
        )

        if swe_data.empty:
            return None, None, None

        swe_data['date'] = pd.to_datetime(swe_data['date'])
        dates = swe_data['date'].values
        swe = swe_data[site_id].values

        # Try to get temperature
        temp = None
        try:
            temp_data = hf.get_point_data(
                dataset="snotel",
                variable="air_temp",
                temporal_resolution="daily",
                aggregation="mean",
                site_ids=site_id,
                date_start=start_date,
                date_end=end_date
            )
            if not temp_data.empty:
                temp = temp_data[site_id].values
        except:
            pass

        return dates, swe, temp

    except Exception as e:
        print(f"    Error fetching SNOTEL: {e}")
        return None, None, None


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(
    forcing: np.ndarray,
    model_output: Dict,
    obs_dates: np.ndarray,
    obs_swe: np.ndarray,
    site: Dict,
    wy: int,
    config_name: str,
    start_date: datetime
) -> ModelMetrics:
    """Compute comprehensive evaluation metrics."""

    n_hours = len(forcing)
    model_dates = np.array([start_date + timedelta(hours=i) for i in range(n_hours)])
    model_swe = model_output['swe']

    # Initialize metrics
    metrics = ModelMetrics(
        site_name=site['name'],
        site_id=site['id'],
        water_year=wy,
        config_name=config_name
    )

    # -------------------------------------------------------------------------
    # Peak SWE metrics
    # -------------------------------------------------------------------------
    metrics.obs_peak_swe = np.nanmax(obs_swe)
    metrics.mod_peak_swe = np.max(model_swe)
    metrics.peak_swe_bias = metrics.mod_peak_swe - metrics.obs_peak_swe
    metrics.peak_swe_bias_pct = 100.0 * metrics.peak_swe_bias / max(metrics.obs_peak_swe, 1.0)

    # Peak dates
    obs_peak_idx = np.nanargmax(obs_swe)
    mod_peak_idx = np.argmax(model_swe)
    metrics.obs_peak_date = pd.Timestamp(obs_dates[obs_peak_idx]).to_pydatetime()
    metrics.mod_peak_date = model_dates[mod_peak_idx]
    metrics.peak_date_bias_days = (metrics.mod_peak_date - metrics.obs_peak_date).days

    # -------------------------------------------------------------------------
    # Melt-out dates
    # -------------------------------------------------------------------------
    # Observed melt-out
    for i in range(obs_peak_idx, len(obs_swe)):
        if obs_swe[i] < 5.0:  # 5mm threshold
            metrics.obs_meltout_date = pd.Timestamp(obs_dates[i]).to_pydatetime()
            break

    # Modeled melt-out
    for i in range(mod_peak_idx, n_hours):
        if model_swe[i] < 1.0:
            metrics.mod_meltout_date = model_dates[i]
            break

    if metrics.obs_meltout_date and metrics.mod_meltout_date:
        metrics.meltout_bias_days = (metrics.mod_meltout_date - metrics.obs_meltout_date).days

    # -------------------------------------------------------------------------
    # Interpolate model to daily for comparison
    # -------------------------------------------------------------------------
    # Get daily model SWE (use noon values)
    model_daily_swe = model_swe[12::24]  # Start at hour 12, every 24 hours
    model_daily_dates = model_dates[12::24]

    # Match to observation dates
    obs_dates_dt = pd.to_datetime(obs_dates)
    model_dates_dt = pd.to_datetime(model_daily_dates)

    # Find common dates
    common_mask_obs = np.isin(obs_dates_dt.date, model_dates_dt.date)
    common_mask_mod = np.isin(model_dates_dt.date, obs_dates_dt.date)

    obs_matched = obs_swe[common_mask_obs]
    mod_matched = model_daily_swe[common_mask_mod]

    # Remove NaN values
    valid = ~np.isnan(obs_matched) & ~np.isnan(mod_matched)
    obs_valid = obs_matched[valid]
    mod_valid = mod_matched[valid]

    if len(obs_valid) > 10:
        # RMSE
        metrics.rmse = np.sqrt(np.mean((mod_valid - obs_valid)**2))

        # MAE
        metrics.mae = np.mean(np.abs(mod_valid - obs_valid))

        # Correlation
        if np.std(obs_valid) > 0 and np.std(mod_valid) > 0:
            metrics.correlation = np.corrcoef(obs_valid, mod_valid)[0, 1]

        # Nash-Sutcliffe Efficiency
        obs_mean = np.mean(obs_valid)
        ss_res = np.sum((obs_valid - mod_valid)**2)
        ss_tot = np.sum((obs_valid - obs_mean)**2)
        if ss_tot > 0:
            metrics.nse = 1 - ss_res / ss_tot

        # Kling-Gupta Efficiency
        r = metrics.correlation if not np.isnan(metrics.correlation) else 0
        alpha = np.std(mod_valid) / np.std(obs_valid) if np.std(obs_valid) > 0 else 1
        beta = np.mean(mod_valid) / np.mean(obs_valid) if np.mean(obs_valid) > 0 else 1
        metrics.kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # -------------------------------------------------------------------------
    # Early season metrics (Oct-Dec)
    # -------------------------------------------------------------------------
    early_end = datetime(wy - 1, 12, 31)

    # Observed early season
    early_obs_mask = obs_dates_dt <= np.datetime64(early_end)
    if np.any(early_obs_mask):
        early_obs = obs_swe[early_obs_mask]
        metrics.early_season_obs_accum = np.nanmax(early_obs) if len(early_obs) > 0 else 0

    # Modeled early season
    early_mod_mask = model_dates <= early_end
    if np.any(early_mod_mask):
        early_mod = model_swe[early_mod_mask]
        metrics.early_season_mod_accum = np.max(early_mod) if len(early_mod) > 0 else 0

    if metrics.early_season_obs_accum > 0:
        metrics.early_season_bias_pct = 100.0 * (
            metrics.early_season_mod_accum - metrics.early_season_obs_accum
        ) / metrics.early_season_obs_accum

    # -------------------------------------------------------------------------
    # Late season RMSE (Apr-Jun)
    # -------------------------------------------------------------------------
    late_start = datetime(wy, 4, 1)
    late_end = datetime(wy, 6, 30)

    late_obs_mask = (obs_dates_dt >= np.datetime64(late_start)) & (obs_dates_dt <= np.datetime64(late_end))
    late_mod_mask = (model_dates_dt >= np.datetime64(late_start)) & (model_dates_dt <= np.datetime64(late_end))

    late_obs = obs_swe[late_obs_mask]
    late_mod = model_daily_swe[late_mod_mask]

    min_len = min(len(late_obs), len(late_mod))
    if min_len > 5:
        late_obs = late_obs[:min_len]
        late_mod = late_mod[:min_len]
        valid_late = ~np.isnan(late_obs)
        if np.sum(valid_late) > 5:
            metrics.late_season_rmse = np.sqrt(np.mean((late_mod[valid_late] - late_obs[valid_late])**2))

    # -------------------------------------------------------------------------
    # Snow partitioning
    # -------------------------------------------------------------------------
    metrics.total_precip = np.sum(forcing[:, 2])
    metrics.total_snowfall = np.sum(model_output['snowfall'])
    metrics.total_rainfall = np.sum(model_output['rainfall'])
    metrics.snow_fraction = metrics.total_snowfall / max(metrics.total_precip, 1.0)

    return metrics


def compute_forcing_bias(
    forcing: np.ndarray,
    obs_temp: Optional[np.ndarray],
    obs_peak_swe: float,
    site_name: str,
    wy: int
) -> ForcingBias:
    """Compute forcing data bias assessment."""

    bias = ForcingBias(site_name=site_name, water_year=wy)

    # CW3E total precip
    bias.cw3e_total_precip = np.sum(forcing[:, 2])
    bias.snotel_peak_swe = obs_peak_swe

    if obs_peak_swe > 0:
        bias.precip_ratio = bias.cw3e_total_precip / obs_peak_swe

    # Temperature comparison (if SNOTEL temp available)
    if obs_temp is not None and len(obs_temp) > 0:
        # Get daily mean model temperature
        model_temp_hourly = forcing[:, 3]
        model_temp_daily = model_temp_hourly.reshape(-1, 24).mean(axis=1)

        # Align lengths
        min_len = min(len(model_temp_daily), len(obs_temp))
        model_t = model_temp_daily[:min_len]
        obs_t = obs_temp[:min_len]

        # Remove NaN
        valid = ~np.isnan(obs_t) & ~np.isnan(model_t)
        if np.sum(valid) > 30:
            model_t = model_t[valid]
            obs_t = obs_t[valid]

            # Convert obs to Kelvin if needed (SNOTEL may be Celsius)
            if np.mean(obs_t) < 100:  # Likely Celsius
                obs_t = obs_t + 273.15

            diff = model_t - obs_t
            bias.temp_bias_mean = np.mean(diff)
            bias.temp_bias_std = np.std(diff)
            bias.temp_rmse = np.sqrt(np.mean(diff**2))
            if np.std(obs_t) > 0:
                bias.temp_corr = np.corrcoef(model_t, obs_t)[0, 1]

    return bias


# =============================================================================
# MODEL RUNNER
# =============================================================================

def run_model(forcing: np.ndarray, config: SnowConfig) -> Dict:
    """Run PySnow model."""
    model = PySnowModel(config)
    return model.run(forcing, dt=3600.0, verbose=False)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_site_year(
    site: Dict,
    wy: int,
    obs_dates: np.ndarray,
    obs_swe: np.ndarray,
    model_results: Dict[str, np.ndarray],  # config_name -> swe array
    start_date: datetime,
    output_dir: Path,
    config_biases: Dict[str, float] = None  # config_name -> bias %
):
    """Create time series plot for a single site-year."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot observations
    ax.plot(obs_dates, obs_swe, 'k-', linewidth=2.5, label='SNOTEL Obs', zorder=10)
    obs_peak = np.nanmax(obs_swe)

    # Generate model dates (hourly)
    n_hours = max(len(swe) for swe in model_results.values()) if model_results else 8760
    model_dates = [start_date + timedelta(hours=i) for i in range(n_hours)]

    # Dynamic color assignment for non-fixed configs
    dynamic_colors = ['tab:purple', 'tab:orange', 'tab:cyan', 'tab:brown']
    dynamic_idx = 0

    # Sort configs: fixed first, then dynamic by bias
    fixed_configs = [c for c in FIXED_PLOT_CONFIGS if c in model_results]
    dynamic_configs = [c for c in model_results if c not in FIXED_PLOT_CONFIGS]
    if config_biases:
        dynamic_configs.sort(key=lambda c: abs(config_biases.get(c, 999)))

    # Plot each config
    for config_name in fixed_configs + dynamic_configs:
        model_swe = model_results[config_name]
        if len(model_swe) == 0:
            continue

        # Assign color
        if config_name in PLOT_COLORS:
            color = PLOT_COLORS[config_name]
        else:
            color = dynamic_colors[dynamic_idx % len(dynamic_colors)]
            dynamic_idx += 1

        # Get bias
        if config_biases and config_name in config_biases:
            bias_pct = config_biases[config_name]
        else:
            mod_peak = np.max(model_swe)
            bias_pct = 100 * (mod_peak - obs_peak) / obs_peak if obs_peak > 0 else 0

        mod_peak = np.max(model_swe)

        # Mark dynamic best performers
        if config_name not in FIXED_PLOT_CONFIGS:
            label = f"* {config_name} ({bias_pct:+.1f}%)"
        else:
            label = f"{config_name} ({bias_pct:+.1f}%)"

        # Use same length as model_swe
        dates_to_plot = model_dates[:len(model_swe)]
        ax.plot(dates_to_plot, model_swe, color=color, linewidth=1.5, alpha=0.85, label=label)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('SWE (mm)', fontsize=12)
    ax.set_title(f"{site['name']} ({site['region']}) - WY{wy}\nObserved Peak: {obs_peak:.0f}mm  |  * = dynamic best performers", fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(start_date, start_date + timedelta(days=365))
    ax.set_ylim(0, None)

    # Save
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    filename = f"{site['name'].replace(' ', '_')}_WY{wy}.png"
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def generate_summary_plots(all_metrics: List, output_dir: Path):
    """Generate summary comparison plots from collected metrics."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Convert metrics to DataFrame for easier analysis
    data = []
    for m in all_metrics:
        data.append({
            'site': m.site_name,
            'water_year': m.water_year,
            'config': m.config_name,
            'peak_bias_pct': m.peak_swe_bias_pct,
            'rmse': m.rmse,
            'nse': m.nse,
            'obs_peak': m.obs_peak_swe,
            'mod_peak': m.mod_peak_swe,
        })
    df = pd.DataFrame(data)

    # 1. Configuration comparison bar chart
    print("  Creating configuration comparison plot...")
    config_stats = df.groupby('config').agg({
        'peak_bias_pct': lambda x: np.abs(x).mean(),
        'rmse': 'mean',
        'nse': 'mean'
    }).reset_index()
    config_stats.columns = ['config', 'mean_abs_bias', 'mean_rmse', 'mean_nse']
    config_stats = config_stats.sort_values('mean_abs_bias')

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    colors = ['tab:blue' if 'Wetbulb' in c else 'tab:red' if 'CLM' in c else 'tab:gray'
              for c in config_stats['config']]

    axes[0].barh(config_stats['config'], config_stats['mean_abs_bias'], color=colors)
    axes[0].set_xlabel('Mean |Bias| (%)', fontsize=11)
    axes[0].set_title('Peak SWE Bias by Configuration', fontsize=12)
    axes[0].axvline(x=20, color='green', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='y', labelsize=8)

    axes[1].barh(config_stats['config'], config_stats['mean_rmse'], color=colors)
    axes[1].set_xlabel('Mean RMSE (mm)', fontsize=11)
    axes[1].set_title('RMSE by Configuration', fontsize=12)
    axes[1].tick_params(axis='y', labelsize=8)

    axes[2].barh(config_stats['config'], config_stats['mean_nse'], color=colors)
    axes[2].set_xlabel('Mean NSE', fontsize=11)
    axes[2].set_title('Nash-Sutcliffe Efficiency', fontsize=12)
    axes[2].axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "config_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Site summary plot - Wetbulb-Tw2C vs CLM-baseline
    print("  Creating site summary plot...")
    sites = df['site'].unique()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, site in enumerate(sites):
        if idx >= len(axes):
            break
        ax = axes[idx]
        site_data = df[df['site'] == site]

        for config, color, marker in [('Wetbulb-Tw2C', 'tab:blue', 'o'),
                                       ('CLM-baseline', 'tab:red', 's')]:
            cfg_data = site_data[site_data['config'] == config].sort_values('water_year')
            if not cfg_data.empty:
                ax.plot(cfg_data['water_year'], cfg_data['peak_bias_pct'],
                       marker=marker, color=color, label=config, linewidth=1.5, markersize=6)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhspan(-20, 20, alpha=0.1, color='green')
        ax.set_xlabel('Water Year')
        ax.set_ylabel('Peak SWE Bias (%)')
        ax.set_title(site, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(sites), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Peak SWE Bias: Wetbulb-Tw2C vs CLM-baseline', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "site_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Threshold sensitivity plot
    print("  Creating threshold sensitivity plot...")
    threshold_configs = ['Wetbulb-Tw0C', 'Wetbulb-Tw1C', 'Wetbulb-Tw2C']
    thresh_data = df[df['config'].isin(threshold_configs)]

    if not thresh_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        config_bias = thresh_data.groupby('config')['peak_bias_pct'].apply(lambda x: np.abs(x).mean())

        thresholds = [0, 1, 2]
        biases = [config_bias.get(f'Wetbulb-Tw{t}C', np.nan) for t in thresholds]

        bars = ax.bar(thresholds, biases, color=['tab:red', 'tab:orange', 'tab:green'], width=0.6)
        ax.set_xlabel('Wet-Bulb Threshold (C)', fontsize=12)
        ax.set_ylabel('Mean |Peak SWE Bias| (%)', fontsize=12)
        ax.set_title('Effect of Wet-Bulb Threshold on Peak SWE Bias', fontsize=14)
        ax.set_xticks(thresholds)

        for i, (t, b) in enumerate(zip(thresholds, biases)):
            if not np.isnan(b):
                ax.text(t, b + 1, f'{b:.1f}%', ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / "threshold_sensitivity.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Regional comparison
    print("  Creating regional comparison plot...")
    site_regions = {s['name']: s['region'] for s in get_snotel_sites()}
    df['region'] = df['site'].map(site_regions)
    regions = df['region'].dropna().unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, region in enumerate(regions):
        if idx >= len(axes):
            break
        ax = axes[idx]
        region_data = df[df['region'] == region]

        config_bias = region_data.groupby('config')['peak_bias_pct'].apply(lambda x: np.abs(x).mean())
        top_configs = config_bias.nsmallest(5).index.tolist()

        for config in top_configs:
            cfg_data = region_data[region_data['config'] == config]
            ax.scatter(cfg_data['water_year'], cfg_data['peak_bias_pct'], label=config, s=50, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhspan(-20, 20, alpha=0.1, color='green')
        ax.set_xlabel('Water Year')
        ax.set_ylabel('Peak SWE Bias (%)')
        ax.set_title(f'{region}', fontsize=12)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    for idx in range(len(regions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Peak SWE Bias by Region (Top 5 Configs)', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "regional_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Summary plots saved to: {plots_dir}")


def generate_site_maps(all_metrics: List, output_dir: Path):
    """
    Generate geographic maps showing site locations colored by mean RMSE.
    Creates 3 maps: best config, baseline CLM, and second-best config.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get site coordinates
    sites = get_snotel_sites()
    site_coords = {s['name']: (s['lat'], s['lon']) for s in sites}
    site_regions = {s['name']: s['region'] for s in sites}

    # Convert metrics to DataFrame
    data = []
    for m in all_metrics:
        data.append({
            'site': m.site_name,
            'config': m.config_name,
            'rmse': m.rmse,
            'peak_bias_pct': m.peak_swe_bias_pct,
            'nse': m.nse,
        })
    df = pd.DataFrame(data)

    # Calculate mean RMSE per site for each config
    site_config_rmse = df.groupby(['site', 'config'])['rmse'].mean().reset_index()

    # Configs to map
    map_configs = [
        ('Wetbulb-Tw2C', 'Best Overall: Wetbulb-Tw2C'),
        ('CLM-baseline', 'Baseline: CLM-baseline'),
        ('Wetbulb-Damp0.7', 'Second Best: Wetbulb-Damp0.7'),
    ]

    # Region colors for markers
    region_markers = {
        'Sierra': 's',       # square
        'Rockies-CO': '^',   # triangle up
        'Rockies-WY': 'v',   # triangle down
        'Wasatch': 'D',      # diamond
        'Cascades': 'o',     # circle
    }

    # Create the 3 maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, (config, title) in enumerate(map_configs):
        ax = axes[ax_idx]

        # Get data for this config
        config_data = site_config_rmse[site_config_rmse['config'] == config]

        # Prepare arrays
        lats = []
        lons = []
        rmses = []
        labels = []
        markers = []

        for _, row in config_data.iterrows():
            site = row['site']
            if site in site_coords:
                lat, lon = site_coords[site]
                lats.append(lat)
                lons.append(lon)
                rmses.append(row['rmse'])
                labels.append(site)
                markers.append(region_markers.get(site_regions.get(site, ''), 'o'))

        lats = np.array(lats)
        lons = np.array(lons)
        rmses = np.array(rmses)

        # Normalize RMSE for colormap (shared scale across all 3 maps)
        vmin, vmax = 100, 400  # Fixed scale for comparison

        # Plot each site with its region marker
        for i in range(len(lats)):
            scatter = ax.scatter(lons[i], lats[i], c=[rmses[i]], cmap='RdYlGn_r',
                               s=300, marker=markers[i], edgecolors='black', linewidth=1.5,
                               vmin=vmin, vmax=vmax, zorder=5)
            # Add site label
            ax.annotate(labels[i], (lons[i], lats[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, fontweight='bold')

        # Add state boundaries approximation (simplified)
        # Western US bounding box
        ax.set_xlim(-125, -103)
        ax.set_ylim(36, 49)

        # Add simple state outlines (approximate)
        state_borders = [
            # CA-NV border (approximate)
            ([-120, -120], [42, 39]),
            ([-120, -114.5], [39, 35.5]),
            # OR-CA border
            ([-124.5, -120], [42, 42]),
            # WA-OR border
            ([-124.5, -117], [46, 46]),
            # ID-WY border
            ([-111, -111], [49, 42]),
            # WY-CO border
            ([-111, -102.5], [41, 41]),
            # UT-CO border
            ([-109, -109], [41, 37]),
        ]
        for xs, ys in state_borders:
            ax.plot(xs, ys, 'k-', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add colorbar for this subplot
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Mean RMSE (mm)', fontsize=10)

    # Add legend for region markers
    legend_elements = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
                                   markersize=10, label=r, markeredgecolor='black')
                       for r, m in region_markers.items()]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=8, title='Region')

    plt.suptitle('SNOTEL Site Performance by Configuration\n(Mean RMSE across all water years)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "site_maps_rmse.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Site maps saved to: {plots_dir / 'site_maps_rmse.png'}")

    # Also create a single detailed map with bias info
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get bias data for best config
    best_config = 'Wetbulb-Tw2C'
    config_bias = df[df['config'] == best_config].groupby('site')['peak_bias_pct'].mean()

    lats = []
    lons = []
    biases = []
    labels = []

    for site in config_bias.index:
        if site in site_coords:
            lat, lon = site_coords[site]
            lats.append(lat)
            lons.append(lon)
            biases.append(config_bias[site])
            labels.append(site)

    lats = np.array(lats)
    lons = np.array(lons)
    biases = np.array(biases)

    # Color by bias (diverging colormap centered at 0)
    vmax_bias = max(abs(biases.min()), abs(biases.max()), 30)
    scatter = ax.scatter(lons, lats, c=biases, cmap='RdBu', s=400,
                        edgecolors='black', linewidth=2, vmin=-vmax_bias, vmax=vmax_bias)

    # Add labels with bias values
    for i in range(len(lats)):
        bias_str = f"{biases[i]:+.1f}%"
        ax.annotate(f"{labels[i]}\n{bias_str}", (lons[i], lats[i]), xytext=(8, 8),
                   textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlim(-125, -103)
    ax.set_ylim(36, 49)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Mean Peak SWE Bias by Site\n{best_config} Configuration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Mean Peak SWE Bias (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(plots_dir / "site_map_bias.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Bias map saved to: {plots_dir / 'site_map_bias.png'}")


# =============================================================================
# REPORTING
# =============================================================================

def generate_comprehensive_report(
    all_metrics: List[ModelMetrics],
    all_forcing_bias: List[ForcingBias],
    output_dir: Path
):
    """Generate comprehensive evaluation report."""

    lines = []
    lines.append("=" * 100)
    lines.append("COMPREHENSIVE PYSNOW SNOTEL EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("Inspired by Ryken et al. (2020) AWR")
    lines.append("=" * 100)
    lines.append("")

    # =========================================================================
    # AGGREGATE METRICS BY CONFIGURATION
    # =========================================================================
    lines.append("AGGREGATE METRICS BY CONFIGURATION")
    lines.append("=" * 100)
    lines.append("")

    # Group by config
    config_stats = {}
    for m in all_metrics:
        if m.config_name not in config_stats:
            config_stats[m.config_name] = {
                'peak_bias_pct': [],
                'rmse': [],
                'nse': [],
                'corr': [],
                'peak_date_bias': [],
                'meltout_bias': [],
                'early_bias_pct': [],
                'n_best': 0,
                'n_total': 0
            }
        stats = config_stats[m.config_name]
        stats['peak_bias_pct'].append(m.peak_swe_bias_pct)
        if not np.isnan(m.rmse):
            stats['rmse'].append(m.rmse)
        if not np.isnan(m.nse):
            stats['nse'].append(m.nse)
        if not np.isnan(m.correlation):
            stats['corr'].append(m.correlation)
        stats['peak_date_bias'].append(m.peak_date_bias_days)
        if m.meltout_bias_days is not None:
            stats['meltout_bias'].append(m.meltout_bias_days)
        if not np.isnan(m.early_season_bias_pct):
            stats['early_bias_pct'].append(m.early_season_bias_pct)
        stats['n_total'] += 1

    # Find best config for each site-year
    site_years = {}
    for m in all_metrics:
        key = (m.site_name, m.water_year)
        if key not in site_years:
            site_years[key] = []
        site_years[key].append(m)

    for key, metrics_list in site_years.items():
        best = min(metrics_list, key=lambda x: abs(x.peak_swe_bias_pct))
        config_stats[best.config_name]['n_best'] += 1

    # Header
    lines.append(f"{'Config':<25} {'Mean Bias%':>10} {'|Bias|%':>10} {'RMSE':>10} {'NSE':>8} {'Corr':>8} "
                f"{'PkDate':>8} {'Melt':>8} {'Early%':>10} {'Best':>6}")
    lines.append("-" * 120)

    # Compute and sort by mean absolute bias
    config_rankings = []
    for config_name, stats in config_stats.items():
        if len(stats['peak_bias_pct']) == 0:
            continue

        mean_bias = np.mean(stats['peak_bias_pct'])
        mean_abs_bias = np.mean(np.abs(stats['peak_bias_pct']))
        mean_rmse = np.mean(stats['rmse']) if stats['rmse'] else np.nan
        mean_nse = np.mean(stats['nse']) if stats['nse'] else np.nan
        mean_corr = np.mean(stats['corr']) if stats['corr'] else np.nan
        mean_pk_date = np.mean(np.abs(stats['peak_date_bias']))
        mean_melt = np.mean(np.abs(stats['meltout_bias'])) if stats['meltout_bias'] else np.nan
        mean_early = np.mean(stats['early_bias_pct']) if stats['early_bias_pct'] else np.nan
        n_best = stats['n_best']

        config_rankings.append((
            config_name, mean_bias, mean_abs_bias, mean_rmse, mean_nse,
            mean_corr, mean_pk_date, mean_melt, mean_early, n_best
        ))

    config_rankings.sort(key=lambda x: x[2])  # Sort by mean absolute bias

    for r in config_rankings:
        lines.append(
            f"{r[0]:<25} {r[1]:>+9.1f}% {r[2]:>9.1f}% {r[3]:>9.1f} {r[4]:>7.2f} {r[5]:>7.2f} "
            f"{r[6]:>7.1f}d {r[7]:>7.1f}d {r[8]:>+9.1f}% {r[9]:>6}"
        )

    # Best overall
    lines.append("")
    best = config_rankings[0]
    lines.append(f">>> BEST OVERALL: {best[0]} <<<")
    lines.append(f"    Mean |Bias|: {best[2]:.1f}%")
    lines.append(f"    Mean RMSE: {best[3]:.1f} mm")
    lines.append(f"    Mean NSE: {best[4]:.2f}")
    lines.append(f"    Times ranked best: {best[9]}")

    # =========================================================================
    # METRICS BY REGION
    # =========================================================================
    lines.append("")
    lines.append("")
    lines.append("METRICS BY REGION")
    lines.append("=" * 100)

    site_to_region = {s['name']: s['region'] for s in get_snotel_sites()}
    regions = set(site_to_region.values())

    for region in sorted(regions):
        lines.append(f"\n{region}:")
        region_metrics = [m for m in all_metrics if site_to_region.get(m.site_name) == region]

        region_config_stats = {}
        for m in region_metrics:
            if m.config_name not in region_config_stats:
                region_config_stats[m.config_name] = []
            region_config_stats[m.config_name].append(abs(m.peak_swe_bias_pct))

        lines.append(f"  {'Config':<25} {'Mean |Bias|%':>15} {'N':>6}")

        region_rankings = []
        for cfg, biases in region_config_stats.items():
            region_rankings.append((cfg, np.mean(biases), len(biases)))
        region_rankings.sort(key=lambda x: x[1])

        for cfg, mean_bias, n in region_rankings[:5]:  # Top 5
            lines.append(f"  {cfg:<25} {mean_bias:>14.1f}% {n:>6}")

        if region_rankings:
            lines.append(f"  Best for {region}: {region_rankings[0][0]} ({region_rankings[0][1]:.1f}%)")

    # =========================================================================
    # FORCING DATA QUALITY ASSESSMENT
    # =========================================================================
    lines.append("")
    lines.append("")
    lines.append("FORCING DATA QUALITY ASSESSMENT (CW3E vs SNOTEL)")
    lines.append("=" * 100)

    # Group by site
    site_bias = {}
    for fb in all_forcing_bias:
        if fb.site_name not in site_bias:
            site_bias[fb.site_name] = []
        site_bias[fb.site_name].append(fb)

    lines.append(f"\n{'Site':<20} {'Years':>6} {'Precip Ratio':>15} {'Temp Bias (K)':>15} {'Assessment':<30}")
    lines.append("-" * 100)

    for site_name, biases in site_bias.items():
        n_years = len(biases)
        precip_ratios = [b.precip_ratio for b in biases if not np.isnan(b.precip_ratio)]
        temp_biases = [b.temp_bias_mean for b in biases if not np.isnan(b.temp_bias_mean)]

        mean_precip_ratio = np.mean(precip_ratios) if precip_ratios else np.nan
        mean_temp_bias = np.mean(temp_biases) if temp_biases else np.nan

        # Assessment
        assessment = []
        if not np.isnan(mean_precip_ratio):
            if mean_precip_ratio < 0.8:
                assessment.append("LOW PRECIP")
            elif mean_precip_ratio > 1.2:
                assessment.append("HIGH PRECIP")
        if not np.isnan(mean_temp_bias):
            if mean_temp_bias > 1.0:
                assessment.append("WARM BIAS")
            elif mean_temp_bias < -1.0:
                assessment.append("COLD BIAS")

        assessment_str = ", ".join(assessment) if assessment else "OK"

        lines.append(
            f"{site_name:<20} {n_years:>6} {mean_precip_ratio:>14.2f}x {mean_temp_bias:>+14.1f} {assessment_str:<30}"
        )

    # =========================================================================
    # DETAILED SITE-YEAR RESULTS
    # =========================================================================
    lines.append("")
    lines.append("")
    lines.append("TOP 3 CONFIGURATIONS PER SITE-YEAR")
    lines.append("=" * 100)

    for (site_name, wy), metrics_list in sorted(site_years.items()):
        sorted_metrics = sorted(metrics_list, key=lambda x: abs(x.peak_swe_bias_pct))[:3]
        lines.append(f"\n{site_name} WY{wy} (Obs Peak: {sorted_metrics[0].obs_peak_swe:.0f}mm):")
        for i, m in enumerate(sorted_metrics, 1):
            lines.append(
                f"  {i}. {m.config_name:<25} Bias: {m.peak_swe_bias_pct:>+6.1f}%, "
                f"RMSE: {m.rmse:>6.1f}mm, NSE: {m.nse:>5.2f}"
            )

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    lines.append("")
    lines.append("")
    lines.append("=" * 100)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 100)

    # Find best config overall and by region
    lines.append(f"\n1. BEST OVERALL CONFIGURATION: {config_rankings[0][0]}")
    lines.append(f"   - Mean |Bias|: {config_rankings[0][2]:.1f}%")
    lines.append(f"   - Mean RMSE: {config_rankings[0][3]:.1f}mm")

    # Check if damping helps
    baseline_configs = [r for r in config_rankings if 'baseline' in r[0]]
    damp_configs = [r for r in config_rankings if 'Damp' in r[0]]

    if baseline_configs and damp_configs:
        baseline_bias = np.mean([r[2] for r in baseline_configs])
        damp_bias = np.mean([r[2] for r in damp_configs])
        if damp_bias < baseline_bias:
            lines.append(f"\n2. DAMPING HELPS: {baseline_bias:.1f}% -> {damp_bias:.1f}% mean |bias|")
        else:
            lines.append(f"\n2. DAMPING NEUTRAL: {baseline_bias:.1f}% vs {damp_bias:.1f}%")

    # Climate-specific recommendations
    lines.append("\n3. CLIMATE-SPECIFIC RECOMMENDATIONS:")
    for region in sorted(regions):
        region_metrics = [m for m in all_metrics if site_to_region.get(m.site_name) == region]
        if not region_metrics:
            continue

        region_config_stats = {}
        for m in region_metrics:
            if m.config_name not in region_config_stats:
                region_config_stats[m.config_name] = []
            region_config_stats[m.config_name].append(abs(m.peak_swe_bias_pct))

        best_cfg = min(region_config_stats.items(), key=lambda x: np.mean(x[1]))
        lines.append(f"   - {region}: {best_cfg[0]} ({np.mean(best_cfg[1]):.1f}%)")

    lines.append("")
    lines.append("=" * 100)

    # Write report
    report_path = output_dir / "comprehensive_evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved: {report_path}")
    print('\n'.join(lines))

    # Also save as CSV for further analysis
    metrics_df = pd.DataFrame([
        {
            'site': m.site_name,
            'water_year': m.water_year,
            'config': m.config_name,
            'obs_peak_swe': m.obs_peak_swe,
            'mod_peak_swe': m.mod_peak_swe,
            'peak_bias_pct': m.peak_swe_bias_pct,
            'peak_date_bias_days': m.peak_date_bias_days,
            'meltout_bias_days': m.meltout_bias_days,
            'rmse': m.rmse,
            'nse': m.nse,
            'correlation': m.correlation,
            'kge': m.kge,
            'early_season_bias_pct': m.early_season_bias_pct,
            'snow_fraction': m.snow_fraction
        }
        for m in all_metrics
    ])
    csv_path = output_dir / "all_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics CSV saved: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run comprehensive SNOTEL evaluation."""

    output_dir = Path(__file__).parent / "comprehensive_evaluation_v2"
    output_dir.mkdir(exist_ok=True)

    sites = get_snotel_sites()
    water_years = get_water_years()
    configs = get_model_configs()

    print("=" * 80)
    print("COMPREHENSIVE PYSNOW SNOTEL EVALUATION")
    print("=" * 80)
    print(f"Sites: {len(sites)}")
    print(f"Water years: {water_years}")
    print(f"Configurations: {len(configs)}")
    print(f"Total potential runs: {len(sites) * len(water_years) * len(configs)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    if not HAS_HYDRODATA:
        print("\nERROR: HydroData not available.")
        return

    all_metrics = []
    all_forcing_bias = []

    for site in sites:
        for wy in water_years:
            print(f"\n{site['name']} ({site['region']}) - WY{wy}")
            print("-" * 60)

            # Get forcing
            print("  Fetching forcing data...")
            forcing = get_forcing_data(site['id'], wy)
            if forcing is None:
                print("  SKIP: Could not get forcing data")
                continue

            # Get observations
            print("  Fetching SNOTEL observations...")
            obs_dates, obs_swe, obs_temp = get_snotel_obs(site['id'], wy)
            if obs_dates is None:
                print("  SKIP: Could not get SNOTEL data")
                continue

            if np.all(np.isnan(obs_swe)) or np.nanmax(obs_swe) < 10:
                print("  SKIP: No valid SWE observations")
                continue

            obs_peak = np.nanmax(obs_swe)
            start_date = datetime(wy - 1, 10, 1)

            # Compute forcing bias
            forcing_bias = compute_forcing_bias(
                forcing, obs_temp, obs_peak, site['name'], wy
            )
            all_forcing_bias.append(forcing_bias)

            # Run all configurations
            best_bias = float('inf')
            best_config = None
            all_swe_results = {}  # Store SWE timeseries for all configs
            config_biases = {}   # Track bias for each config

            for config_name, config in configs.items():
                try:
                    output = run_model(forcing, config)

                    metrics = compute_metrics(
                        forcing, output, obs_dates, obs_swe,
                        site, wy, config_name, start_date
                    )
                    all_metrics.append(metrics)

                    if abs(metrics.peak_swe_bias_pct) < abs(best_bias):
                        best_bias = metrics.peak_swe_bias_pct
                        best_config = config_name

                    # Store SWE and bias for all configs
                    all_swe_results[config_name] = output['swe']
                    config_biases[config_name] = metrics.peak_swe_bias_pct

                except Exception as e:
                    print(f"    ERROR ({config_name}): {e}")

            print(f"  Obs Peak: {obs_peak:.0f}mm | Best: {best_config} ({best_bias:+.1f}%)")

            # Determine which configs to plot:
            # 1-3: Fixed configs (Wetbulb-Tw2C, Wetbulb-Damp0.7, CLM-baseline)
            # 4-5: Top 2 dynamic performers not in fixed list
            plot_results = {}

            # Add fixed configs
            for cfg in FIXED_PLOT_CONFIGS:
                if cfg in all_swe_results:
                    plot_results[cfg] = all_swe_results[cfg]

            # Find top 2 dynamic performers (not in fixed list)
            dynamic_candidates = [
                (cfg, abs(bias)) for cfg, bias in config_biases.items()
                if cfg not in FIXED_PLOT_CONFIGS
            ]
            dynamic_candidates.sort(key=lambda x: x[1])  # Sort by |bias|

            for i, (cfg, _) in enumerate(dynamic_candidates[:2]):
                plot_results[cfg] = all_swe_results[cfg]

            # Create time series plot for this site-year
            if plot_results:
                try:
                    plot_path = plot_site_year(
                        site, wy, obs_dates, obs_swe, plot_results,
                        start_date, output_dir, config_biases
                    )
                    print(f"  Plot saved: {plot_path.name}")
                except Exception as e:
                    print(f"  Plot error: {e}")

    # Generate report and summary plots
    if all_metrics:
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)
        generate_comprehensive_report(all_metrics, all_forcing_bias, output_dir)

        print("\n" + "=" * 80)
        print("GENERATING SUMMARY PLOTS")
        print("=" * 80)
        generate_summary_plots(all_metrics, output_dir)

        print("\n" + "=" * 80)
        print("GENERATING SITE MAPS")
        print("=" * 80)
        generate_site_maps(all_metrics, output_dir)
    else:
        print("\nNo results to summarize.")


if __name__ == '__main__':
    main()
