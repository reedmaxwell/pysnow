#!/usr/bin/env python3
"""
Run PySnow model for any SNOTEL site and water year.

Fetches forcing data from CW3E via HydroData and compares to SNOTEL SWE observations.

Usage:
    python run_snotel_site.py --site "Css Lab" --state CA --wy 2024
    python run_snotel_site.py --site_id "428:CA:SNTL" --wy 2024
    python run_snotel_site.py --list-sites --state CA

Examples:
    # Run for CSS Lab, Water Year 2024
    python run_snotel_site.py --site "Css Lab" --state CA --wy 2024

    # Run for a site by ID
    python run_snotel_site.py --site_id "1050:WY:SNTL" --wy 2023

    # List all SNOTEL sites in a state
    python run_snotel_site.py --list-sites --state WY
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import hf_hydrodata as hf
    import subsettools as st
    HAS_HYDRODATA = True
except ImportError:
    HAS_HYDRODATA = False
    print("Warning: hf_hydrodata or subsettools not available")
    print("Install with: pip install hf_hydrodata subsettools")

from pysnow import PySnowModel, SnowConfig


def register_hydrodata():
    """Register HydroData API using environment variables.

    Set these environment variables before running:
        export HF_EMAIL="your_email@example.com"
        export HF_PIN="your_pin"

    Or register at: https://hydrogen.princeton.edu/pin
    """
    if HAS_HYDRODATA:
        email = os.environ.get('HF_EMAIL')
        pin = os.environ.get('HF_PIN')

        if email and pin:
            hf.register_api_pin(email, pin)
        else:
            print("Warning: HF_EMAIL and HF_PIN environment variables not set.")
            print("Register at https://hydrogen.princeton.edu/pin")
            print("Then: export HF_EMAIL='your_email' && export HF_PIN='your_pin'")


def list_snotel_sites(state):
    """List all SNOTEL sites in a state."""
    if not HAS_HYDRODATA:
        print("Error: hf_hydrodata required")
        return None

    register_hydrodata()

    metadata = hf.get_point_metadata(
        dataset="snotel",
        variable="swe",
        temporal_resolution="daily",
        aggregation="sod",
        state=state
    )

    print(f"\nSNOTEL Sites in {state}:")
    print("-" * 80)
    print(f"{'Site ID':<20} {'Site Name':<30} {'Lat':>10} {'Lon':>12} {'Elev (m)':>10}")
    print("-" * 80)

    for _, row in metadata.iterrows():
        elev = row.get('elevation_m', row.get('elevation', 'N/A'))
        print(f"{row['site_id']:<20} {row['site_name']:<30} {row['latitude']:>10.4f} {row['longitude']:>12.4f} {elev:>10}")

    return metadata


def get_site_info(site_name=None, site_id=None, state=None):
    """Get site metadata by name or ID."""
    if not HAS_HYDRODATA:
        raise RuntimeError("hf_hydrodata required")

    register_hydrodata()

    if site_id:
        # Extract state from site_id if not provided
        if state is None:
            parts = site_id.split(':')
            if len(parts) >= 2:
                state = parts[1]

    if state is None:
        raise ValueError("Must provide state or site_id with state code")

    metadata = hf.get_point_metadata(
        dataset="snotel",
        variable="swe",
        temporal_resolution="daily",
        aggregation="sod",
        state=state
    )

    if site_id:
        site_info = metadata[metadata['site_id'] == site_id]
    else:
        site_info = metadata[metadata['site_name'] == site_name]

    if len(site_info) == 0:
        raise ValueError(f"Site not found: {site_name or site_id}")

    return site_info.iloc[0]


def get_snotel_swe(site_id, start, end):
    """Fetch SNOTEL SWE observations."""
    if not HAS_HYDRODATA:
        return None, None

    register_hydrodata()

    try:
        data = hf.get_point_data(
            dataset="snotel",
            variable="swe",
            temporal_resolution="daily",
            aggregation="sod",
            site_ids=site_id,
            date_start=start,
            date_end=end
        )
        data['date'] = pd.to_datetime(data['date'])

        # Find the SWE column (site_id column)
        swe_col = [c for c in data.columns if c != 'date'][0]
        return data['date'].values, data[swe_col].values
    except Exception as e:
        print(f"Warning: Could not fetch SNOTEL data: {e}")
        return None, None


def get_forcing_data(lat, lon, start, end, precip_factor=1.0):
    """Fetch forcing data from CW3E via HydroData."""
    if not HAS_HYDRODATA:
        raise RuntimeError("hf_hydrodata required")

    register_hydrodata()

    # Define domain bounds for single point
    latlon_bounds = [[lat, lon], [lat, lon]]
    bounds, mask = st.define_latlon_domain(latlon_bounds=latlon_bounds, grid="conus2")

    print(f"Fetching forcing data for ({lat:.4f}, {lon:.4f})...")
    print(f"Grid bounds: {bounds}")

    dataset = "CW3E"
    version = "1.0"

    # Fetch all forcing variables
    variables = {
        'DSWR': 'downward_shortwave',
        'DLWR': 'downward_longwave',
        'APCP': 'precipitation',
        'Temp': 'air_temp',
        'UGRD': 'east_windspeed',
        'VGRD': 'north_windspeed',
        'Press': 'atmospheric_pressure',
        'SPFH': 'specific_humidity',
    }

    data = {}
    for name, var in variables.items():
        print(f"  Loading {name}...")
        options = {
            "dataset": dataset,
            "period": "hourly",
            "variable": var,
            "start_time": start,
            "end_time": end,
            "grid_bounds": bounds,
            "dataset_version": version
        }
        data[name] = hf.get_gridded_data(options)[:, 0, 0]

    # Clean up precipitation
    data['APCP'][data['APCP'] < 0] = 0.0
    data['APCP'] = data['APCP'] * precip_factor

    # Combine into forcing array
    # Columns: [sw_down, lw_down, precip, t_air, wind_u, wind_v, pressure, q_air]
    # Note: precip is in mm/s from CW3E, need to convert to mm/hr for model
    forcing = np.column_stack([
        data['DSWR'],
        data['DLWR'],
        data['APCP'] * 3600.0,  # mm/s -> mm/hr
        data['Temp'],
        data['UGRD'],
        data['VGRD'],
        data['Press'],
        data['SPFH'],
    ])

    print(f"Loaded {len(forcing)} hours of forcing")
    print(f"  Temp range: {data['Temp'].min()-273.15:.1f} to {data['Temp'].max()-273.15:.1f} C")
    print(f"  Precip total: {data['APCP'].sum() * 3600:.1f} mm")

    return forcing


def run_configurations(forcing, configs):
    """Run model with multiple configurations."""
    results = {}
    for name, config in configs.items():
        print(f"  Running {name}...")
        model = PySnowModel(config)
        results[name] = model.run(forcing, dt=3600.0, verbose=False)
    return results


def create_comparison_plot(dates, results, snotel_dates, snotel_swe, site_name, wy, output_path):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot 1: Full year SWE
    ax1 = axes[0]
    if snotel_dates is not None:
        ax1.plot(snotel_dates, snotel_swe, 'k-', linewidth=2.5, label='SNOTEL', zorder=10)

    for (name, output), color in zip(results.items(), colors):
        ax1.plot(dates, output['swe'], color=color, linewidth=1.5, label=name, alpha=0.8)

    ax1.set_ylabel('SWE (mm)')
    ax1.set_title(f'PySnow Model Comparison: {site_name} WY{wy}')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Early season (Oct-Feb)
    ax2 = axes[1]
    early_end = datetime(wy - 1, 10, 1) + timedelta(days=120)  # ~Feb 1

    if snotel_dates is not None:
        mask = snotel_dates < np.datetime64(early_end)
        if mask.any():
            ax2.plot(snotel_dates[mask], snotel_swe[mask], 'k-', linewidth=2.5, label='SNOTEL', zorder=10)

    for (name, output), color in zip(results.items(), colors):
        mask = [d < early_end for d in dates]
        ax2.plot([d for d, m in zip(dates, mask) if m],
                 output['swe'][:sum(mask)],
                 color=color, linewidth=1.5, label=name, alpha=0.8)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('SWE (mm)')
    ax2.set_title('Early Season (Oct-Jan)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()


def print_summary(results, dates, snotel_swe):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY: Peak SWE and Melt-out")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Peak SWE':>12} {'Peak Date':>14} {'Melt-out':>14}")
    print("-" * 80)

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
        print(f"{name:<25} {peak_swe:>10.1f} mm {peak_date.strftime('%Y-%m-%d'):>14} {meltout_str:>14}")

    if snotel_swe is not None:
        peak_swe = np.nanmax(snotel_swe)
        peak_idx = np.nanargmax(snotel_swe)
        # Find melt-out in SNOTEL
        meltout_idx = None
        for i in range(peak_idx, len(snotel_swe)):
            if snotel_swe[i] < 1:
                meltout_idx = i
                break
        print(f"{'SNOTEL':<25} {peak_swe:>10.1f} mm {'(observed)':>14}")


def get_default_configs(thin_snow_damping=0.0):
    """Return default model configurations to compare."""
    return {
        # CLM-like baseline
        'CLM-linear': SnowConfig(
            rain_snow_method='linear',
            rain_snow_params={'t_all_snow': 273.15 + 1.0, 't_all_rain': 273.15 + 3.0},
            albedo_method='clm_decay',
            phase_change_method='clm',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
            thin_snow_damping=thin_snow_damping,
        ),

        # Jennings humidity-aware
        'Jennings': SnowConfig(
            rain_snow_method='jennings',
            rain_snow_params={},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
            thin_snow_damping=thin_snow_damping,
        ),

        # Dai (2008) hyperbolic tangent
        'Dai': SnowConfig(
            rain_snow_method='dai',
            rain_snow_params={},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
            thin_snow_damping=thin_snow_damping,
        ),

        # Wetbulb threshold (Wang et al. 2019)
        'Wetbulb': SnowConfig(
            rain_snow_method='wetbulb',
            rain_snow_params={'tw_threshold': 273.15 + 1.0},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
            thin_snow_damping=thin_snow_damping,
        ),

        # Wetbulb sigmoid (Wang et al. 2019)
        'Wetbulb-Sigmoid': SnowConfig(
            rain_snow_method='wetbulb_sigmoid',
            rain_snow_params={},
            albedo_method='snicar_lite',
            albedo_params={'dr_wet': 80.0},
            phase_change_method='hierarchy',
            ground_flux_const=2.0,
            t_soil=273.15 + 5.0,
            thin_snow_damping=thin_snow_damping,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run PySnow model for any SNOTEL site and water year.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--site', type=str, help='SNOTEL site name (e.g., "Css Lab")')
    parser.add_argument('--site_id', type=str, help='SNOTEL site ID (e.g., "428:CA:SNTL")')
    parser.add_argument('--state', type=str, help='State code (e.g., CA, WY, CO)')
    parser.add_argument('--wy', type=int, help='Water year (e.g., 2024 = Oct 2023 - Sep 2024)')
    parser.add_argument('--list-sites', action='store_true', help='List all SNOTEL sites in state')
    parser.add_argument('--precip-factor', type=float, default=1.0,
                        help='Precipitation adjustment factor (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots and forcing files')
    parser.add_argument('--save-forcing', action='store_true',
                        help='Save forcing data to text file')
    parser.add_argument('--thin-snow-damping', type=float, default=0.5,
                        help='Thin snow energy damping factor [0-1] (default: 0.5)')

    args = parser.parse_args()

    # List sites mode
    if args.list_sites:
        if not args.state:
            print("Error: --state required with --list-sites")
            sys.exit(1)
        list_snotel_sites(args.state)
        return

    # Validate inputs
    if not args.site and not args.site_id:
        print("Error: Must provide --site or --site_id")
        parser.print_help()
        sys.exit(1)

    if not args.wy:
        print("Error: Must provide --wy (water year)")
        sys.exit(1)

    # Get site info
    print("\n" + "=" * 60)
    print("PySnow SNOTEL Site Runner")
    print("=" * 60)

    site_info = get_site_info(
        site_name=args.site,
        site_id=args.site_id,
        state=args.state
    )

    site_name = site_info['site_name']
    site_id = site_info['site_id']
    lat = site_info['latitude']
    lon = site_info['longitude']

    print(f"\nSite: {site_name} ({site_id})")
    print(f"Location: {lat:.4f}, {lon:.4f}")
    print(f"Water Year: {args.wy}")

    # Define date range
    start = f"{args.wy - 1}-10-01"
    end = f"{args.wy}-10-01"

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'sensitivity_study'
    output_dir.mkdir(exist_ok=True)

    # Fetch forcing data
    print("\n" + "-" * 60)
    forcing = get_forcing_data(lat, lon, start, end, precip_factor=args.precip_factor)

    # Save forcing if requested
    if args.save_forcing:
        forcing_file = output_dir / f'forcing1D.{site_name.replace(" ", "_")}.{start}-{end}.txt'
        np.savetxt(forcing_file, forcing, fmt='%.8f')
        print(f"Forcing saved: {forcing_file}")

    # Time axis
    n_hours = len(forcing)
    start_dt = datetime(args.wy - 1, 10, 1)
    dates = [start_dt + timedelta(hours=i) for i in range(n_hours)]

    # Fetch SNOTEL observations
    print("\n" + "-" * 60)
    print("Fetching SNOTEL observations...")
    snotel_dates, snotel_swe = get_snotel_swe(site_id, start, end)

    if snotel_swe is not None:
        print(f"  Peak observed SWE: {np.nanmax(snotel_swe):.1f} mm")

    # Run model configurations
    print("\n" + "-" * 60)
    print("Running model configurations...")
    configs = get_default_configs(thin_snow_damping=args.thin_snow_damping)
    results = run_configurations(forcing, configs)

    # Create plot
    plot_file = output_dir / f'pysnow_{site_name.replace(" ", "_")}_WY{args.wy}.png'
    create_comparison_plot(dates, results, snotel_dates, snotel_swe, site_name, args.wy, plot_file)

    # Print summary
    print_summary(results, dates, snotel_swe)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
