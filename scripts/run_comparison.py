#!/usr/bin/env python3
"""
Run PySnow model with different parameterization combinations.
Compare to ParFlow-CLM results and SNOTEL observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pysnow import PySnowModel, SnowConfig

# Try to load SNOTEL data
try:
    import hf_hydrodata as hf
    email = os.environ.get('HF_EMAIL')
    pin = os.environ.get('HF_PIN')
    if email and pin:
        hf.register_api_pin(email, pin)
        HAS_SNOTEL = True
    else:
        HAS_SNOTEL = False
        print("Warning: Set HF_EMAIL and HF_PIN environment variables for SNOTEL data")
except:
    HAS_SNOTEL = False
    print("Warning: hf_hydrodata not available")


def load_forcing(filepath):
    """Load forcing file in CLM format."""
    data = np.loadtxt(filepath)
    # Columns: DSWR DLWR APCP Temp Wind-U Wind-V Press SPFH
    # APCP is in mm/s (kg/m²/s), convert to mm/hr
    data[:, 2] = data[:, 2] * 3600.0  # mm/s -> mm/hr
    return data


def get_snotel_data():
    """Fetch SNOTEL SWE data."""
    if not HAS_SNOTEL:
        return None, None

    import pandas as pd
    try:
        site_id = "428:CA:SNTL"
        data = hf.get_point_data(
            dataset="snotel", variable='swe', temporal_resolution='daily', aggregation='sod',
            site_ids=site_id, date_start="2023-10-01", date_end="2024-10-01"
        )
        data['date'] = pd.to_datetime(data['date'])
        return data['date'].values, data['428:CA:SNTL'].values
    except Exception as e:
        print(f"Warning: Could not fetch SNOTEL data: {e}")
        return None, None


def run_configuration(forcing, config, name):
    """Run model with given configuration."""
    model = PySnowModel(config)
    outputs = model.run(forcing, dt=3600.0, verbose=False)
    return outputs


def main():
    # Configuration
    base_dir = Path(__file__).parent.parent
    forcing_file = base_dir / 'forcing1D.Css_Lab.2023-10-01-2024-10-01.txt'  # Original forcing

    print("Loading forcing data...")
    forcing = load_forcing(forcing_file)
    n_hours = len(forcing)
    print(f"Loaded {n_hours} hours of forcing")

    # Time axis
    start = datetime(2023, 10, 1)
    dates = [start + timedelta(hours=i) for i in range(n_hours)]

    # Define configurations to test
    configs = {
        # Baseline (CLM-like)
        'CLM-like': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='clm_decay',
            albedo_params={'conn': 0.2, 'tau': 86400},
            phase_change_method='clm',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # VIC albedo (late season fix)
        'VIC-albedo': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='vic',
            albedo_params={},
            phase_change_method='clm',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # SNOWCLIM hierarchy (early season fix)
        'SNOWCLIM-hierarchy': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='essery',
            albedo_params={},
            phase_change_method='hierarchy',
            cold_content_tax=True,
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # Combined best
        'Combined': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='vic',
            albedo_params={'thaw_a': 0.80, 'thaw_b': 0.50},
            phase_change_method='hierarchy',
            cold_content_tax=True,
            ground_flux_const=1.0,
            t_soil=273.15 + 3.0,
        ),

        # SNICAR-lite albedo
        'SNICAR-lite': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # Reduced ground flux
        'Reduced-G': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='vic',
            phase_change_method='hierarchy',
            ground_flux_const=0.5,
            t_soil=273.15 + 2.0,
        ),

        # Wetbulb partitioning (Wang et al. 2019)
        'Wetbulb': SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 273.15 + 1.0},  # 1°C threshold
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # Dai (2008) hyperbolic tangent
        'Dai': SnowConfig(
            rain_snow_method='dai',
            rain_snow_params={},  # Uses default coefficients
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),

        # Wang et al. (2019) wetbulb sigmoid
        'Wetbulb-Sigmoid': SnowConfig(
            rain_snow_method='wetbulb_sigmoid',
            rain_snow_params={},  # Uses default coefficients
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
        ),
    }

    # Run all configurations
    print("\nRunning configurations...")
    results = {}
    for name, config in configs.items():
        print(f"  {name}...")
        results[name] = run_configuration(forcing, config, name)

    # Load SNOTEL data
    print("\nLoading SNOTEL data...")
    snotel_dates, snotel_swe = get_snotel_data()

    # Create comparison plot
    print("\nCreating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Plot 1: Full year SWE
    ax1 = axes[0]
    if snotel_dates is not None:
        ax1.plot(snotel_dates, snotel_swe, 'k-', linewidth=2.5, label='SNOTEL', zorder=10)

    for (name, output), color in zip(results.items(), colors):
        ax1.plot(dates, output['swe'], color=color, linewidth=1.5, label=name, alpha=0.8)

    ax1.set_ylabel('SWE (mm)')
    ax1.set_title('PySnow Model Comparison: Full Year')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([datetime(2023, 10, 1), datetime(2024, 8, 1)])

    # Plot 2: Early season
    ax2 = axes[1]
    early_end = datetime(2024, 2, 1)

    if snotel_dates is not None:
        mask = snotel_dates < np.datetime64(early_end)
        ax2.plot(snotel_dates[mask], snotel_swe[mask], 'k-', linewidth=2.5, label='SNOTEL', zorder=10)

    for (name, output), color in zip(results.items(), colors):
        mask = [d < early_end for d in dates]
        ax2.plot([d for d, m in zip(dates, mask) if m],
                 output['swe'][:sum(mask)],
                 color=color, linewidth=1.5, label=name, alpha=0.8)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('SWE (mm)')
    ax2.set_title('PySnow Model Comparison: Early Season (Oct-Jan)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([datetime(2023, 10, 1), early_end])

    plt.tight_layout()

    # Save plot
    plot_path = base_dir / 'sensitivity_study' / 'pysnow_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Peak SWE and Melt-out")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Peak SWE':>12} {'Peak Date':>12} {'Melt-out':>12}")
    print("-" * 70)

    for name, output in results.items():
        peak_swe = np.max(output['swe'])
        peak_idx = np.argmax(output['swe'])
        peak_date = dates[peak_idx]

        # Find melt-out
        meltout = None
        for i in range(peak_idx, len(output['swe'])):
            if output['swe'][i] < 1:
                meltout = dates[i]
                break

        meltout_str = meltout.strftime('%Y-%m-%d') if meltout else 'N/A'
        print(f"{name:<25} {peak_swe:>10.1f} mm {peak_date.strftime('%Y-%m-%d'):>12} {meltout_str:>12}")

    if snotel_swe is not None:
        peak_swe = np.nanmax(snotel_swe)
        print(f"{'SNOTEL':<25} {peak_swe:>10.1f} mm {'2024-04-02':>12} {'2024-05-21':>12}")

    plt.close()


if __name__ == '__main__':
    main()
