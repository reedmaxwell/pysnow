# PySnow

A modular Python snow model with switchable parameterizations for rapid testing and development.

PySnow combines snow physics from CLM (Community Land Model), CoLM, and SNOWCLIM into a flexible framework where different parameterization schemes can be mixed and matched.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pysnow.git
cd pysnow

# Install dependencies
pip install numpy matplotlib

# Install package (optional)
pip install -e .
```

## Quick Start

```python
from pysnow import PySnowModel, SnowConfig

# Configure the model
config = SnowConfig(
    rain_snow_method='linear',
    albedo_method='vic',
    phase_change_method='hierarchy'
)

# Create model instance
model = PySnowModel(config)

# Run for multiple timesteps
outputs = model.run(forcing_data, dt=3600.0)

# Access results
swe = outputs['swe']
melt = outputs['melt']
albedo = outputs['albedo']
```

## Configuration Options

### Rain-Snow Partitioning (`rain_snow_method`)

| Method | Description | Reference |
|--------|-------------|-----------|
| `threshold` | Simple temperature threshold | - |
| `linear` | Linear ramp between thresholds | CLM default |
| `jennings` | Humidity-dependent logistic | Jennings et al. 2018 |
| `wet_bulb` | Wet-bulb temperature threshold | CoLM |

```python
config = SnowConfig(
    rain_snow_method='linear',
    rain_snow_params={
        't_all_snow': 274.15,  # 100% snow below this [K]
        't_all_rain': 276.15   # 100% rain above this [K]
    }
)
```

### Albedo (`albedo_method`)

| Method | Description | Reference |
|--------|-------------|-----------|
| `constant` | Fixed albedo value | - |
| `clm_decay` | Exponential decay with age | CLM |
| `vic` | Separate cold/thaw decay | VIC |
| `essery` | New snow refresh | Essery et al. 2013 |
| `tarboton` | Temperature-dependent aging | UEB |
| `snicar_lite` | Grain size evolution | Simplified SNICAR |

```python
config = SnowConfig(
    albedo_method='vic',
    albedo_params={
        'albedo_new': 0.85,
        'albedo_min': 0.5,
        'accum_a': 0.94,  # Cold decay coefficient
        'thaw_a': 0.82    # Thaw decay coefficient
    }
)
```

### Phase Change (`phase_change_method`)

| Method | Description | Reference |
|--------|-------------|-----------|
| `hierarchy` | Cold content → refreeze → melt | SNOWCLIM |
| `clm` | Simultaneous processes | CLM |

```python
config = SnowConfig(
    phase_change_method='hierarchy',
    cold_content_tax=True  # Apply SNOWCLIM cold content limitation
)
```

### Other Parameters

```python
config = SnowConfig(
    # Thermal conductivity
    conductivity_method='jordan',  # 'jordan', 'sturm', 'calonne'

    # Compaction
    compaction_method='essery',    # 'essery', 'anderson', 'simple'

    # Ground heat flux
    ground_flux_method='constant',
    ground_flux_const=2.0,         # W/m²
    t_soil=278.15,                 # Soil temperature [K]

    # Turbulent fluxes
    stability_correction=True,
    z_wind=10.0,                   # Wind measurement height [m]
    z_temp=2.0,                    # Temperature measurement height [m]
    z_0=0.001,                     # Roughness length [m]

    # Liquid water
    max_water_frac=0.10,           # Maximum liquid water fraction
    drain_rate=2.778e-5            # Drainage rate [m/s]
)
```

## Forcing Data Format

PySnow expects forcing data as a numpy array with shape `(n_timesteps, 8)`:

| Column | Variable | Units |
|--------|----------|-------|
| 0 | Shortwave radiation (down) | W/m² |
| 1 | Longwave radiation (down) | W/m² |
| 2 | Precipitation rate | mm/hr |
| 3 | Air temperature | K |
| 4 | Wind U-component | m/s |
| 5 | Wind V-component | m/s |
| 6 | Air pressure | Pa |
| 7 | Specific humidity | kg/kg |

## Output Variables

The `model.run()` method returns a dictionary with:

| Key | Description | Units |
|-----|-------------|-------|
| `swe` | Snow water equivalent | mm |
| `depth` | Snow depth | m |
| `t_snow` | Snow temperature | K |
| `albedo` | Surface albedo | - |
| `density` | Bulk snow density | kg/m³ |
| `melt` | Melt rate | mm/timestep |
| `runoff` | Runoff | mm/timestep |
| `snowfall` | Snowfall | mm/timestep |
| `rainfall` | Rainfall | mm/timestep |
| `sublimation` | Sublimation (negative=loss) | mm/timestep |
| `Rn` | Net radiation | W/m² |
| `H` | Sensible heat flux | W/m² |
| `LE` | Latent heat flux | W/m² |
| `G` | Ground heat flux | W/m² |

## Single Timestep Mode

For coupling to other models or custom time loops:

```python
model = PySnowModel(config)

for i in range(n_timesteps):
    forcing = {
        'sw_down': sw_data[i],
        'lw_down': lw_data[i],
        'precip': precip_data[i],      # mm/hr
        't_air': temp_data[i],
        'wind': wind_data[i],
        'pressure': press_data[i],
        'q_air': humidity_data[i]
    }

    state = model.step(forcing, dt=3600.0)

    # Access state variables
    print(f"SWE: {state.swe:.1f} mm")
    print(f"Melt: {model.fluxes.melt:.2f} mm")
```

## Example: Compare Configurations

```python
import numpy as np
from pysnow import PySnowModel, SnowConfig

# Load forcing data
forcing = np.loadtxt('forcing.txt')

# Define configurations to test
configs = {
    'CLM-like': SnowConfig(
        albedo_method='clm_decay',
        phase_change_method='clm'
    ),
    'SNOWCLIM': SnowConfig(
        albedo_method='essery',
        phase_change_method='hierarchy',
        cold_content_tax=True
    ),
    'VIC-albedo': SnowConfig(
        albedo_method='vic',
        phase_change_method='hierarchy'
    )
}

# Run all configurations
results = {}
for name, config in configs.items():
    model = PySnowModel(config)
    results[name] = model.run(forcing, dt=3600.0)
    print(f"{name}: Peak SWE = {np.max(results[name]['swe']):.1f} mm")
```

## Module Structure

```
pysnow/
├── __init__.py       # Package exports
├── constants.py      # Physical constants
├── rain_snow.py      # Rain-snow partitioning
├── albedo.py         # Albedo schemes
├── thermal.py        # Thermal properties & phase change
├── compaction.py     # Density evolution
├── turbulent.py      # Turbulent heat fluxes
├── radiation.py      # Radiation balance
├── model.py          # Main model driver
└── docs/
    └── CLM_SNOW_TUNING_SUMMARY.md
```

## Physical Constants

Key constants defined in `constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `TFRZ` | 273.15 K | Freezing point |
| `LF_FUSION` | 333,500 J/kg | Latent heat of fusion |
| `CP_ICE` | 2,090 J/(kg·K) | Heat capacity of ice |
| `RHO_ICE` | 917 kg/m³ | Density of ice |
| `RHO_WATER` | 1,000 kg/m³ | Density of water |

## References

- Anderson, E.A. (1976). A point energy and mass balance model of a snow cover. NOAA Technical Report NWS 19.
- Essery, R., et al. (2013). A comparison of four different methods of snow model calibration. Hydrological Processes.
- Jennings, K.S., et al. (2018). Spatial variation of the rain-snow temperature threshold across the Northern Hemisphere. Nature Communications.
- Jordan, R. (1991). A one-dimensional temperature model for a snow cover. CRREL Special Report 91-16.
- Oleson, K.W., et al. (2013). Technical description of version 4.5 of the Community Land Model (CLM). NCAR Technical Note.

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.

## Citation

If you use PySnow in your research, please cite:

```
PySnow: A modular Python snow model for parameterization testing.
https://github.com/yourusername/pysnow
```
