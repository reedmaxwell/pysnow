#!/usr/bin/env python3
"""
Generate site maps from existing comprehensive evaluation metrics.
Uses the all_metrics.csv from the previous run.

NOTE: Use conda environment 'subsettools' to run this script!
  source /Users/reed/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools && python generate_maps_only.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shapefile  # pyshp

# Site coordinates (same as in comprehensive_snotel_test_v2.py)
SITE_COORDS = {
    "CSS Lab": (39.31, -120.37),
    "Leavitt Lake": (38.28, -119.56),
    "Berthoud Summit": (39.80, -105.78),
    "Loveland Basin": (39.68, -105.88),
    "Canyon": (44.73, -110.50),
    "Togwotee Pass": (43.76, -110.07),
    "Trial Lake": (40.68, -110.95),
    "Paradise": (46.79, -121.74),
}

SITE_REGIONS = {
    "CSS Lab": "Sierra",
    "Leavitt Lake": "Sierra",
    "Berthoud Summit": "Rockies-CO",
    "Loveland Basin": "Rockies-CO",
    "Canyon": "Rockies-WY",
    "Togwotee Pass": "Rockies-WY",
    "Trial Lake": "Wasatch",
    "Paradise": "Cascades",
}

REGION_MARKERS = {
    'Sierra': 's',       # square
    'Rockies-CO': '^',   # triangle up
    'Rockies-WY': 'v',   # triangle down
    'Wasatch': 'D',      # diamond
    'Cascades': 'o',     # circle
}

# Path to geodata
GEODATA_DIR = Path("/Users/reed/Projects/ParFlow_example_cases/single_column_SWE/pysnow/geodata")
STATES_SHP = GEODATA_DIR / "cb_2021_us_state_20m.shp"
HUC2_SHP = Path("/Users/reed/reed_papers/Quantity_Accessibility_GW_NAmerica/maps/Watershed_Boundary_Dataset_HUC_2s_-4080625415220690096/HU02.shp")

# Western US states to include
WESTERN_STATES = ['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM']

# Western US HUC2 codes (to filter the shapefile)
WESTERN_HUC2 = ['10', '11', '13', '14', '15', '16', '17', '18']


def draw_states(ax, xlim, ylim):
    """Draw state boundaries from shapefile."""
    if not STATES_SHP.exists():
        print(f"Warning: States shapefile not found at {STATES_SHP}")
        return

    sf = shapefile.Reader(str(STATES_SHP))

    for shape_rec in sf.shapeRecords():
        # Get state abbreviation
        state_abbr = shape_rec.record['STUSPS']

        # Only draw western states
        if state_abbr not in WESTERN_STATES:
            continue

        # Get the shape
        shape = shape_rec.shape

        # Handle multi-part polygons
        parts = list(shape.parts) + [len(shape.points)]

        for i in range(len(parts) - 1):
            start = parts[i]
            end = parts[i + 1]
            points = shape.points[start:end]

            if len(points) < 3:
                continue

            lons = [p[0] for p in points]
            lats = [p[1] for p in points]

            # Filter to view extent (with buffer)
            if max(lons) < xlim[0] - 5 or min(lons) > xlim[1] + 5:
                continue
            if max(lats) < ylim[0] - 5 or min(lats) > ylim[1] + 5:
                continue

            ax.plot(lons, lats, color='gray', linewidth=0.5, alpha=0.4, zorder=1)


def draw_huc2_boundaries(ax, xlim, ylim):
    """Draw HUC2 watershed boundaries from shapefile."""
    if not HUC2_SHP.exists():
        print(f"Warning: HUC2 shapefile not found at {HUC2_SHP}")
        return

    sf = shapefile.Reader(str(HUC2_SHP))

    # Find the HUC2 field name
    field_names = [field[0] for field in sf.fields[1:]]
    huc2_field = None
    for name in ['huc2', 'HUC2', 'HUC_2', 'huc_2', 'HU_2_Code']:
        if name in field_names:
            huc2_field = name
            break

    for shape_rec in sf.shapeRecords():
        # Filter to western HUC2s if we can identify them
        if huc2_field:
            huc2_code = str(shape_rec.record[huc2_field])
            if huc2_code not in WESTERN_HUC2:
                continue

        shape = shape_rec.shape

        # Handle multi-part polygons
        parts = list(shape.parts) + [len(shape.points)]

        for i in range(len(parts) - 1):
            start = parts[i]
            end = parts[i + 1]
            points = shape.points[start:end]

            if len(points) < 3:
                continue

            lons = [p[0] for p in points]
            lats = [p[1] for p in points]

            # Filter to view extent (with buffer)
            if max(lons) < xlim[0] - 2 or min(lons) > xlim[1] + 2:
                continue
            if max(lats) < ylim[0] - 2 or min(lats) > ylim[1] + 2:
                continue

            ax.plot(lons, lats, color='black', linewidth=1.0, alpha=0.6,
                    linestyle='-', zorder=2)


def generate_site_maps(df: pd.DataFrame, output_dir: Path):
    """Generate geographic maps showing site locations colored by mean RMSE."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Calculate mean RMSE per site for each config
    site_config_rmse = df.groupby(['site', 'config'])['rmse'].mean().reset_index()

    # Configs to map
    map_configs = [
        ('Wetbulb-Tw2C', 'Best Overall: Wetbulb-Tw2C'),
        ('CLM-baseline', 'Baseline: CLM-baseline'),
        ('Wetbulb-Damp0.7', 'Second Best: Wetbulb-Damp0.7'),
    ]

    # Create the 3 maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, (config, title) in enumerate(map_configs):
        ax = axes[ax_idx]

        # Set limits first
        xlim = (-125, -103)
        ylim = (36, 49)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Draw geographic features
        draw_states(ax, xlim, ylim)
        draw_huc2_boundaries(ax, xlim, ylim)

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
            if site in SITE_COORDS:
                lat, lon = SITE_COORDS[site]
                lats.append(lat)
                lons.append(lon)
                rmses.append(row['rmse'])
                labels.append(site)
                markers.append(REGION_MARKERS.get(SITE_REGIONS.get(site, ''), 'o'))

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
                       textcoords='offset points', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')

        # Add colorbar for this subplot
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Mean RMSE (mm)', fontsize=10)

    # Add legend for region markers
    legend_elements = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
                                   markersize=10, label=r, markeredgecolor='black')
                       for r, m in REGION_MARKERS.items()]
    # Add HUC2 line to legend
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=1.0,
                                       alpha=0.6, label='HUC2 Divide'))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=0.5,
                                       alpha=0.4, label='State Border'))
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=7, title='Legend')

    plt.suptitle('SNOTEL Site Performance by Configuration\n(Mean RMSE across all water years)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "site_maps_rmse.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Site maps saved to: {plots_dir / 'site_maps_rmse.png'}")

    # Create a single detailed map with bias info
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set limits first
    xlim = (-125, -103)
    ylim = (36, 49)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Draw geographic features
    draw_states(ax, xlim, ylim)
    draw_huc2_boundaries(ax, xlim, ylim)

    # Get bias data for best config
    best_config = 'Wetbulb-Tw2C'
    config_bias = df[df['config'] == best_config].groupby('site')['peak_bias_pct'].mean()

    lats = []
    lons = []
    biases = []
    labels = []

    for site in config_bias.index:
        if site in SITE_COORDS:
            lat, lon = SITE_COORDS[site]
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
                        edgecolors='black', linewidth=2, vmin=-vmax_bias, vmax=vmax_bias,
                        zorder=5)

    # Add labels with bias values
    for i in range(len(lats)):
        bias_str = f"{biases[i]:+.1f}%"
        ax.annotate(f"{labels[i]}\n{bias_str}", (lons[i], lats[i]), xytext=(8, 8),
                   textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=6)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Mean Peak SWE Bias by Site\n{best_config} Configuration', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Mean Peak SWE Bias (%)', fontsize=11)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=1.0, alpha=0.6, label='HUC2 Divide'),
        plt.Line2D([0], [0], color='gray', linewidth=0.5, alpha=0.4, label='State Border'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "site_map_bias.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Bias map saved to: {plots_dir / 'site_map_bias.png'}")


def main():
    # Load existing metrics
    output_dir = Path("/Users/reed/Projects/ParFlow_example_cases/single_column_SWE/pysnow/comprehensive_evaluation_v2")
    csv_path = output_dir / "all_metrics.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    print(f"Loading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows")
    print(f"Sites: {df['site'].unique()}")
    print(f"Configs: {df['config'].nunique()}")

    generate_site_maps(df, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
