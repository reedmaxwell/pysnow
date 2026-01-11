#!/usr/bin/env python3
"""
Diagnose early season snow behavior.

Examines forcing data and model behavior to understand why thin snow
melts too fast in early season.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import hf_hydrodata as hf
    import subsettools as st
    HAS_HYDRODATA = True
except ImportError:
    HAS_HYDRODATA = False

from pysnow import PySnowModel, SnowConfig
from pysnow.run_snotel_site import (
    register_hydrodata, get_site_info, get_snotel_swe, get_forcing_data
)


def diagnose_site(site_id, wy, early_season_days=120):
    """
    Diagnose early season behavior for a site.
    """
    register_hydrodata()

    # Get site info
    site_info = get_site_info(site_id=site_id)
    site_name = site_info['site_name']
    lat = site_info['latitude']
    lon = site_info['longitude']

    print(f"\nDiagnosing: {site_name} ({site_id})")
    print(f"Location: {lat:.4f}, {lon:.4f}")
    print(f"Water Year: {wy}")

    # Date range
    start = f"{wy - 1}-10-01"
    end = f"{wy}-10-01"

    # Get forcing
    forcing = get_forcing_data(lat, lon, start, end)

    # Get SNOTEL
    snotel_dates, snotel_swe = get_snotel_swe(site_id, start, end)

    # Time axis
    n_hours = len(forcing)
    start_dt = datetime(wy - 1, 10, 1)
    dates = [start_dt + timedelta(hours=i) for i in range(n_hours)]

    # Early season subset
    early_hours = early_season_days * 24

    # Extract forcing variables
    sw_down = forcing[:early_hours, 0]
    lw_down = forcing[:early_hours, 1]
    precip = forcing[:early_hours, 2]  # mm/hr
    t_air = forcing[:early_hours, 3] - 273.15  # Convert to C
    wind = np.sqrt(forcing[:early_hours, 4]**2 + forcing[:early_hours, 5]**2)
    pressure = forcing[:early_hours, 6]
    q_air = forcing[:early_hours, 7]

    early_dates = dates[:early_hours]

    # Test different ground heat flux values
    gflux_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    configs = {}
    for gf in gflux_values:
        configs[f'G={gf}'] = SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 273.15 + 1.0},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=gf,
            t_soil=273.15 + 2.0,  # Colder soil
        )

    # Also test depth-dependent ground flux
    configs['G=depth-dep'] = SnowConfig(
        rain_snow_method='wetbulb',
        rain_snow_params={'tw_threshold': 273.15 + 1.0},
        albedo_method='snicar_lite',
        albedo_params={'dr_wet': 80.0},
        phase_change_method='hierarchy',
        ground_flux_method='snow_depth',  # Will need to implement this
        ground_flux_const=2.0,
        t_soil=273.15 + 2.0,
    )

    # Run models
    results = {}
    for name, config in configs.items():
        model = PySnowModel(config)
        try:
            results[name] = model.run(forcing[:early_hours], dt=3600.0)
        except Exception as e:
            print(f"  {name} failed: {e}")

    # Create diagnostic plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Temperature and precipitation
    ax1 = axes[0]
    ax1b = ax1.twinx()

    # Daily averages for clarity
    days = len(early_dates) // 24
    daily_dates = [early_dates[i*24] for i in range(days)]
    daily_temp = [np.mean(t_air[i*24:(i+1)*24]) for i in range(days)]
    daily_precip = [np.sum(precip[i*24:(i+1)*24]) for i in range(days)]

    ax1.bar(daily_dates, daily_precip, width=0.8, alpha=0.5, color='blue', label='Precip (mm/day)')
    ax1b.plot(daily_dates, daily_temp, 'r-', linewidth=1.5, label='Temp (°C)')
    ax1b.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1b.axhline(1, color='orange', linestyle='--', alpha=0.5, label='1°C threshold')

    ax1.set_ylabel('Precipitation (mm/day)', color='blue')
    ax1b.set_ylabel('Temperature (°C)', color='red')
    ax1.set_title(f'Early Season Forcing: {site_name} WY{wy}')
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')

    # Plot 2: SWE comparison
    ax2 = axes[1]

    if snotel_dates is not None:
        # Subset SNOTEL to early season
        early_end = np.datetime64(early_dates[-1])
        snotel_mask = snotel_dates <= early_end
        ax2.plot(snotel_dates[snotel_mask], snotel_swe[snotel_mask],
                 'k-', linewidth=2.5, label='SNOTEL')

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (name, output), color in zip(results.items(), colors):
        ax2.plot(early_dates, output['swe'], color=color, linewidth=1.5,
                 label=name, alpha=0.8)

    ax2.set_ylabel('SWE (mm)')
    ax2.set_title('Ground Heat Flux Sensitivity')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Ground heat flux and melt
    ax3 = axes[2]

    for (name, output), color in zip(results.items(), colors):
        if 'G' in output:
            daily_G = [np.mean(output['G'][i*24:(i+1)*24]) for i in range(days)]
            ax3.plot(daily_dates, daily_G, color=color, linewidth=1.5,
                     label=f'{name} G', alpha=0.8)

    ax3.set_ylabel('Ground Heat Flux (W/m²)')
    ax3.set_title('Ground Heat Flux')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative snowfall vs SWE
    ax4 = axes[3]

    # Calculate cumulative snowfall from wetbulb method
    from pysnow.rain_snow import partition_rain_snow, calc_rh_from_specific_humidity

    cum_snow = np.zeros(early_hours)
    for i in range(early_hours):
        if precip[i] > 0:
            rh = calc_rh_from_specific_humidity(q_air[i], t_air[i] + 273.15, pressure[i])
            snow_frac = partition_rain_snow(
                precip[i], t_air[i] + 273.15, rh=rh, pressure=pressure[i],
                method='wetbulb', tw_threshold=273.15 + 1.0
            )
            if i > 0:
                cum_snow[i] = cum_snow[i-1] + precip[i] * snow_frac
            else:
                cum_snow[i] = precip[i] * snow_frac
        elif i > 0:
            cum_snow[i] = cum_snow[i-1]

    ax4.plot(early_dates, cum_snow, 'b-', linewidth=2, label='Cumulative Snowfall')

    if snotel_dates is not None:
        ax4.plot(snotel_dates[snotel_mask], snotel_swe[snotel_mask],
                 'k-', linewidth=2.5, label='SNOTEL SWE')

    # Best model result
    best_name = 'G=0.5' if 'G=0.5' in results else list(results.keys())[0]
    if best_name in results:
        ax4.plot(early_dates, results[best_name]['swe'], 'g-', linewidth=1.5,
                 label=f'Model ({best_name})', alpha=0.8)

    ax4.set_ylabel('SWE / Cum. Snowfall (mm)')
    ax4.set_xlabel('Date')
    ax4.set_title('Cumulative Snowfall vs SWE')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'sensitivity_study'
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / f'diagnose_{site_name.replace(" ", "_")}_WY{wy}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved: {plot_file}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Early Season Summary (first 120 days)")
    print("=" * 60)

    total_precip = np.sum(precip)
    warm_precip = np.sum(precip[t_air > 1.0])
    cold_precip = np.sum(precip[t_air <= 1.0])

    print(f"Total precipitation: {total_precip:.1f} mm")
    print(f"  Cold (T <= 1°C): {cold_precip:.1f} mm ({100*cold_precip/total_precip:.1f}%)")
    print(f"  Warm (T > 1°C): {warm_precip:.1f} mm ({100*warm_precip/total_precip:.1f}%)")
    print(f"Mean temperature: {np.mean(t_air):.1f}°C")
    print(f"Hours below freezing: {np.sum(t_air < 0)} / {early_hours}")

    if snotel_swe is not None:
        snotel_early = snotel_swe[snotel_mask]
        print(f"\nSNOTEL SWE at day 120: {snotel_early[-1]:.1f} mm")

    print(f"\nModel SWE at day 120:")
    for name, output in results.items():
        print(f"  {name}: {output['swe'][early_hours-1]:.1f} mm")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose early season snow behavior')
    parser.add_argument('--site_id', type=str, required=True, help='SNOTEL site ID')
    parser.add_argument('--wy', type=int, required=True, help='Water year')
    parser.add_argument('--days', type=int, default=120, help='Early season days to analyze')

    args = parser.parse_args()

    diagnose_site(args.site_id, args.wy, args.days)
