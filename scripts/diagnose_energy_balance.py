#!/usr/bin/env python3
"""
Detailed energy balance diagnostic for thin snow melt events.
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
    HAS_HYDRODATA = True
except ImportError:
    HAS_HYDRODATA = False

from pysnow import PySnowModel, SnowConfig
from pysnow.run_snotel_site import (
    register_hydrodata, get_site_info, get_snotel_swe, get_forcing_data
)


def detailed_energy_diagnostic(site_id, wy, start_day=0, end_day=60):
    """
    Track detailed energy balance during early season.
    """
    register_hydrodata()

    # Get site info
    site_info = get_site_info(site_id=site_id)
    site_name = site_info['site_name']
    lat = site_info['latitude']
    lon = site_info['longitude']

    print(f"\nEnergy Balance Diagnostic: {site_name}")
    print(f"Days {start_day} to {end_day} of WY{wy}")

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

    # Subset to analysis period
    start_hour = start_day * 24
    end_hour = end_day * 24
    forcing_sub = forcing[start_hour:end_hour]
    dates_sub = dates[start_hour:end_hour]

    # Run model with detailed output
    config = SnowConfig(
        rain_snow_method='wetbulb',
        rain_snow_params={'tw_threshold': 273.15 + 1.0},
        albedo_method='snicar_lite',
        albedo_params={'dr_wet': 80.0},
        phase_change_method='hierarchy',
        ground_flux_const=2.0,
        t_soil=273.15 + 2.0,
    )

    model = PySnowModel(config)
    outputs = model.run(forcing_sub, dt=3600.0)

    # Extract energy balance terms
    Rn = outputs['Rn']
    H = outputs['H']
    LE = outputs['LE']
    G = outputs['G']
    swe = outputs['swe']
    melt = outputs['melt']
    sublim = outputs['sublimation']
    snowfall = outputs['snowfall']

    # Calculate daily totals
    n_days = len(dates_sub) // 24
    daily_dates = [dates_sub[i*24] for i in range(n_days)]

    def daily_mean(arr):
        return [np.mean(arr[i*24:(i+1)*24]) for i in range(n_days)]

    def daily_sum(arr):
        return [np.sum(arr[i*24:(i+1)*24]) for i in range(n_days)]

    daily_Rn = daily_mean(Rn)
    daily_H = daily_mean(H)
    daily_LE = daily_mean(LE)
    daily_G = daily_mean(G)
    daily_swe = [swe[(i+1)*24-1] for i in range(n_days)]  # End of day SWE
    daily_melt = daily_sum(melt)
    daily_sublim = daily_sum(sublim)
    daily_snow = daily_sum(snowfall)
    daily_temp = daily_mean(forcing_sub[:, 3] - 273.15)

    # Net energy
    daily_Qnet = [Rn + H + LE + G for Rn, H, LE, G in zip(daily_Rn, daily_H, daily_LE, daily_G)]

    # Create plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    # Plot 1: Energy balance components
    ax1 = axes[0]
    ax1.plot(daily_dates, daily_Rn, 'y-', linewidth=2, label='Rn (Net Rad)')
    ax1.plot(daily_dates, daily_H, 'r-', linewidth=2, label='H (Sensible)')
    ax1.plot(daily_dates, daily_LE, 'b-', linewidth=2, label='LE (Latent)')
    ax1.plot(daily_dates, daily_G, 'brown', linewidth=2, label='G (Ground)')
    ax1.plot(daily_dates, daily_Qnet, 'k--', linewidth=2, label='Q_net')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_ylabel('Energy Flux (W/m²)')
    ax1.set_title(f'Energy Balance: {site_name} WY{wy} (Days {start_day}-{end_day})')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Temperature
    ax2 = axes[1]
    ax2.fill_between(daily_dates, daily_temp, 0, where=[t > 0 for t in daily_temp],
                     alpha=0.3, color='red', label='T > 0°C')
    ax2.fill_between(daily_dates, daily_temp, 0, where=[t <= 0 for t in daily_temp],
                     alpha=0.3, color='blue', label='T <= 0°C')
    ax2.plot(daily_dates, daily_temp, 'k-', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axhline(1, color='orange', linestyle='--', alpha=0.5, label='1°C')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mass balance
    ax3 = axes[2]
    ax3.bar(daily_dates, daily_snow, width=0.8, alpha=0.7, color='blue', label='Snowfall')
    ax3.bar(daily_dates, [-m for m in daily_melt], width=0.8, alpha=0.7, color='red', label='Melt')
    ax3.bar(daily_dates, daily_sublim, width=0.8, alpha=0.7, color='purple',
            bottom=[-m for m in daily_melt], label='Sublimation')
    ax3.axhline(0, color='gray', linestyle='-')
    ax3.set_ylabel('Mass Flux (mm/day)')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: SWE comparison
    ax4 = axes[3]
    ax4.plot(daily_dates, daily_swe, 'g-', linewidth=2, label='Model SWE')

    if snotel_dates is not None:
        # Subset SNOTEL
        snotel_start = np.datetime64(dates_sub[0])
        snotel_end = np.datetime64(dates_sub[-1])
        snotel_mask = (snotel_dates >= snotel_start) & (snotel_dates <= snotel_end)
        if snotel_mask.any():
            ax4.plot(snotel_dates[snotel_mask], snotel_swe[snotel_mask],
                     'k-', linewidth=2.5, label='SNOTEL')

    ax4.set_ylabel('SWE (mm)')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Cumulative comparison
    ax5 = axes[4]
    cum_snow = np.cumsum(daily_snow)
    cum_melt = np.cumsum(daily_melt)
    cum_sublim = np.cumsum([-s for s in daily_sublim])  # Make positive for loss

    ax5.plot(daily_dates, cum_snow, 'b-', linewidth=2, label='Cum. Snowfall')
    ax5.plot(daily_dates, cum_melt, 'r-', linewidth=2, label='Cum. Melt')
    ax5.plot(daily_dates, cum_sublim, 'purple', linewidth=2, label='Cum. Sublim Loss')
    ax5.plot(daily_dates, daily_swe, 'g--', linewidth=2, label='SWE')

    ax5.set_ylabel('Cumulative (mm)')
    ax5.set_xlabel('Date')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'sensitivity_study'
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / f'energy_balance_{site_name.replace(" ", "_")}_WY{wy}_d{start_day}-{end_day}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Energy Balance Summary")
    print("=" * 60)

    print(f"\nDaily mean fluxes (W/m²):")
    print(f"  Net radiation (Rn): {np.mean(daily_Rn):.1f}")
    print(f"  Sensible heat (H):  {np.mean(daily_H):.1f}")
    print(f"  Latent heat (LE):   {np.mean(daily_LE):.1f}")
    print(f"  Ground heat (G):    {np.mean(daily_G):.1f}")
    print(f"  Net energy (Q):     {np.mean(daily_Qnet):.1f}")

    print(f"\nMass balance totals (mm):")
    print(f"  Snowfall:     {sum(daily_snow):.1f}")
    print(f"  Melt:         {sum(daily_melt):.1f}")
    print(f"  Sublimation:  {sum(daily_sublim):.1f}")
    print(f"  Final SWE:    {daily_swe[-1]:.1f}")

    # Identify melt events
    print("\nMelt events (days with melt > 5mm):")
    for i, (d, m, t) in enumerate(zip(daily_dates, daily_melt, daily_temp)):
        if m > 5:
            print(f"  Day {start_day + i}: {d.strftime('%Y-%m-%d')}, Melt={m:.1f}mm, T={t:.1f}°C")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detailed energy balance diagnostic')
    parser.add_argument('--site_id', type=str, required=True, help='SNOTEL site ID')
    parser.add_argument('--wy', type=int, required=True, help='Water year')
    parser.add_argument('--start-day', type=int, default=0, help='Start day of analysis')
    parser.add_argument('--end-day', type=int, default=60, help='End day of analysis')

    args = parser.parse_args()

    detailed_energy_diagnostic(args.site_id, args.wy, args.start_day, args.end_day)
