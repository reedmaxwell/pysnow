# PySnow: A Modular Snow Model

## Overview

PySnow is a standalone, modular snow model with switchable parameterizations for each physical process. It is designed for research applications where comparing different snow physics representations is important.

**Key Features:**
- Modular design with interchangeable schemes for each process
- Single-layer bulk snowpack representation
- Hourly timestep capability
- Energy balance approach with full radiation and turbulent flux calculations
- Multiple rain-snow partitioning methods including wet-bulb temperature

**Repository:** [github.com/reedmaxwell/pysnow](https://github.com/reedmaxwell/pysnow)

---

## Model Structure

### State Variables

| Variable | Symbol | Units | Description |
|----------|--------|-------|-------------|
| Snow Water Equivalent | SWE | mm | Total water mass in snowpack |
| Snow Depth | $d$ | m | Physical depth of snowpack |
| Bulk Density | $\rho$ | kg/m³ | Mass per unit volume |
| Snow Temperature | $T_s$ | K | Bulk temperature |
| Albedo | $\alpha$ | - | Shortwave reflectivity |
| Snow Age | $\tau$ | s | Time since last snowfall |
| Liquid Water | $W_l$ | mm | Liquid water content |
| Grain Radius | $r_g$ | μm | Effective optical grain size (SNICAR only) |

### Energy Balance

The surface energy balance is:

$$Q_{net} = R_n + H + LE + G$$

Where:
- $R_n$ = Net radiation (shortwave + longwave)
- $H$ = Sensible heat flux
- $LE$ = Latent heat flux
- $G$ = Ground heat flux

Sign convention: Positive toward the snowpack surface.

---

## Physical Processes and Equations

### 1. Rain-Snow Partitioning

Determines the fraction of precipitation falling as snow vs. rain.

#### Available Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `threshold` | Binary temperature threshold | - |
| `linear` | Linear ramp (CLM-style) | CLM Technical Note |
| `jennings` | Humidity-dependent logistic | Jennings et al. (2018) |
| `dai` | Hyperbolic tangent | Dai (2008) |
| `wetbulb` | Wet-bulb temperature threshold | Wang et al. (2019) |
| `wetbulb_linear` | Wet-bulb with linear transition | - |
| `wetbulb_sigmoid` | Wet-bulb sigmoid probability | Wang et al. (2019) |

#### Equations

**Linear Ramp (CLM):**
$$f_s = \begin{cases} 1 & T < T_{snow} \\ \frac{T_{rain} - T}{T_{rain} - T_{snow}} & T_{snow} \leq T \leq T_{rain} \\ 0 & T > T_{rain} \end{cases}$$

Default: $T_{snow} = 274.15$ K (1°C), $T_{rain} = 276.15$ K (3°C)

**Jennings Humidity-Dependent:**
$$P(snow) = \frac{1}{1 + \exp(a + b \cdot T_C + g \cdot RH)}$$

Where $T_C$ is temperature in Celsius, $RH$ is relative humidity (%), and default coefficients are $a = -10.04$, $b = 1.41$, $g = 0.09$.

**Wet-Bulb Temperature (Wang et al. 2019):**

Uses wet-bulb temperature $T_w$ instead of air temperature:
$$f_s = \begin{cases} 1 & T_w \leq T_{w,threshold} \\ 0 & T_w > T_{w,threshold} \end{cases}$$

Wet-bulb temperature calculated using Stull (2011) approximation:
$$T_w = T \cdot \arctan[0.151977(RH + 8.313659)^{0.5}] + \arctan(T + RH) - \arctan(RH - 1.676331) + 0.00391838 \cdot RH^{1.5} \cdot \arctan(0.023101 \cdot RH) - 4.686035$$

**Recommended:** `wetbulb` with $T_{w,threshold} = 275.15$ K (2°C) based on comprehensive SNOTEL evaluation.

---

### 2. Albedo

Controls shortwave radiation absorption and is critical for energy balance.

#### Available Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `constant` | Fixed value | - |
| `clm_decay` | Exponential decay with age | CLM Technical Note |
| `vic` | Separate cold/warm decay rates | VIC Model |
| `essery` | Fresh snow refresh + decay | Essery et al. (2013) |
| `tarboton` | Temperature-dependent aging | Tarboton & Luce (1996) |
| `snicar_lite` | Grain size evolution | Flanner & Zender (2006) |

#### Equations

**VIC Model:**
$$\alpha = \alpha_{new} \cdot \begin{cases} a_{accum}^{\tau_d^{b_{accum}}} & T_s < T_{frz} \\ a_{thaw}^{\tau_d^{b_{thaw}}} & T_s \geq T_{frz} \end{cases}$$

Where $\tau_d$ is snow age in days. Default parameters:
- Accumulating (cold): $a_{accum} = 0.94$, $b_{accum} = 0.58$
- Thawing (warm): $a_{thaw} = 0.82$, $b_{thaw} = 0.46$

**Essery (2013):**
$$\frac{d\alpha}{dt} = \begin{cases} -1/\tau_a & T_s < T_{melt} \text{ (cold)} \\ -(\alpha - \alpha_{min})/\tau_m & T_s \geq T_{melt} \text{ (melting)} \end{cases}$$

With fresh snow refresh:
$$\alpha_{new} = \alpha + (\alpha_{max} - \alpha) \cdot \min(1, S/S_0)$$

**SNICAR-lite (grain size evolution):**
$$\frac{dr_g}{dt} = \begin{cases} dr_{wet} & T_s \geq T_{frz} - 0.5 \\ dr_{dry} & T_s < T_{frz} - 0.5 \end{cases}$$

$$\alpha = \alpha_{ref} - 0.12 \cdot \log_2(r_g / 100)$$

**Thin Snow Correction:**
$$\alpha_{eff} = f \cdot \alpha_{snow} + (1-f) \cdot \alpha_{ground}$$

Where $f = \min(1, d/z_{crit})$ with $z_{crit} = 0.02$ m.

---

### 3. Thermal Conductivity

Determines heat transfer through the snowpack.

#### Available Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `jordan` | CLM/CoLM default | Jordan (1991) |
| `sturm` | Seasonal snow empirical | Sturm et al. (1997) |
| `calonne` | Microstructure-based | Calonne et al. (2011) |

#### Equations

**Jordan (1991):**
$$k = k_{air} + (7.75 \times 10^{-5} \rho + 1.105 \times 10^{-6} \rho^2)(k_{ice} - k_{air})$$

Where $k_{air} = 0.023$ W/(m·K), $k_{ice} = 2.29$ W/(m·K).

**Sturm et al. (1997):**
$$k = \begin{cases} 0.023 + 0.234(\rho/1000) & \rho < 156 \text{ kg/m}^3 \\ 0.138 - 1.01(\rho/1000) + 3.233(\rho/1000)^2 & \rho \geq 156 \text{ kg/m}^3 \end{cases}$$

---

### 4. Phase Change

Handles melt and refreeze processes.

#### Available Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `hierarchy` | Energy hierarchy (CC → refreeze → melt) | SNOWCLIM |
| `clm` | CLM-style simultaneous | CLM Technical Note |

#### Energy Hierarchy Method

1. **Cold Content Warming:** If $Q > 0$ and $T_s < T_{frz}$:
   - Energy to warm to 0°C: $Q_{warm} = -SWE \cdot c_p \cdot (T_s - T_{frz})$

2. **Refreezing:** If $Q < 0$ and liquid water exists:
   - Energy released: $Q_{refreeze} = m_{refreeze} \cdot L_f$

3. **Melting:** If $Q > 0$ and $T_s = T_{frz}$:
   - Melt rate: $m_{melt} = Q / (L_f \cdot \rho_w)$

Where $L_f = 333,500$ J/kg (latent heat of fusion).

**Cold Content:**
$$CC = SWE \cdot c_p \cdot (T_s - T_{frz})$$

**Thin Snow Damping:**
For $SWE < SWE_{threshold}$:
$$Q_{eff} = Q \cdot (1 - \gamma \cdot (1 - SWE/SWE_{threshold}))$$

Where $\gamma$ is the damping factor (0-1).

---

### 5. Compaction

Controls snow density evolution over time.

#### Available Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| `essery` | Viscosity-based | Essery et al. (2013) |
| `anderson` | CLM-style | Anderson (1976) |
| `simple` | Linear approach to max density | - |

#### Equations

**Essery Viscosity-Based:**
$$\frac{d\rho}{dt} = \rho \left( \frac{\rho \cdot m \cdot g}{\eta} + c_1 \exp(-c_2 T_{neg} - c_3(\rho - \rho_0)) \right)$$

Viscosity:
$$\eta = \eta_0 \exp(c_4 T_{neg} + c_5 \rho)$$

Where $T_{neg} = \max(0, T_{frz} - T_s)$.

**Fresh Snow Density (Anderson 1976):**
$$\rho_{new} = \begin{cases} 169.15 & T_C > 2 \\ 50 + 1.7(T_C + 15)^{1.5} & -15 < T_C \leq 2 \\ 50 & T_C \leq -15 \end{cases}$$

---

### 6. Turbulent Fluxes

Sensible and latent heat exchange with atmosphere.

#### Equations

**Bulk Aerodynamic Formulation:**
$$H = \rho_a c_p C_H U (T_a - T_s)$$
$$LE = \rho_a L C_H U (q_a - q_s)$$

Neutral exchange coefficient:
$$C_H = \frac{\kappa^2}{\ln(z_w/z_0) \cdot \ln(z_T/z_h)}$$

**Stability Correction (Louis 1979):**

Bulk Richardson number:
$$Ri_b = \frac{g \cdot z \cdot (T_a - T_s)}{T_{mean} \cdot U^2}$$

Stability function:
$$f_H = \begin{cases} 1 - \frac{3cRi_b}{1 + 3c^2 C_H \sqrt{-Ri_b \cdot z/z_0}} & Ri_b < 0 \text{ (unstable)} \\ \frac{1}{1 + 2cRi_b/\sqrt{1 + Ri_b}} & Ri_b > 0 \text{ (stable)} \end{cases}$$

**Latent Heat:**
$$L = \begin{cases} L_s = 2.834 \times 10^6 \text{ J/kg} & T_s < T_{frz} \text{ (sublimation)} \\ L_v = 2.501 \times 10^6 \text{ J/kg} & T_s \geq T_{frz} \text{ (evaporation)} \end{cases}$$

---

### 7. Radiation

#### Net Radiation
$$R_n = (1 - \alpha) SW_{down} + \epsilon (LW_{down} - \sigma T_s^4)$$

Where:
- $\alpha$ = surface albedo
- $\epsilon$ = snow emissivity (0.98)
- $\sigma$ = Stefan-Boltzmann constant ($5.67 \times 10^{-8}$ W/(m²·K⁴))

---

### 8. Ground Heat Flux

#### Methods
- **Constant:** $G = G_{const}$ (default 2.0 W/m²)
- **Gradient:** $G = k_{interface} \cdot (T_{soil} - T_s) / (d/2 + d_{soil})$

---

## Configuration

### SnowConfig Parameters

```python
@dataclass
class SnowConfig:
    # Rain-snow partitioning
    rain_snow_method: str = 'linear'      # Method name
    rain_snow_params: Dict = {}           # Method-specific parameters

    # Albedo
    albedo_method: str = 'vic'            # Method name
    albedo_params: Dict = {}              # Method-specific parameters

    # Thermal conductivity
    conductivity_method: str = 'jordan'   # jordan, sturm, calonne

    # Phase change
    phase_change_method: str = 'hierarchy'  # hierarchy, clm
    cold_content_tax: bool = False        # SNOWCLIM cold content tax
    thin_snow_damping: float = 0.0        # Damping factor [0-1]
    thin_snow_threshold: float = 50.0     # SWE threshold [mm]

    # Compaction
    compaction_method: str = 'essery'     # essery, anderson, simple

    # Turbulent fluxes
    stability_correction: bool = True
    windless_coeff: float = 0.0           # W/(m²·K)
    z_wind: float = 10.0                  # Wind measurement height [m]
    z_temp: float = 2.0                   # Temperature measurement height [m]
    z_0: float = 0.001                    # Roughness length [m]

    # Ground heat flux
    ground_flux_method: str = 'constant'
    ground_flux_const: float = 2.0        # W/m²
    t_soil: float = 278.15                # Soil temperature [K]

    # Liquid water
    min_water_frac: float = 0.01          # Irreducible water fraction
    max_water_frac: float = 0.10          # Maximum water fraction
    drain_rate: float = 2.778e-5          # Drainage rate [m/s]
```

### Recommended Configuration

Based on comprehensive evaluation across 8 SNOTEL sites:

```python
config = SnowConfig(
    rain_snow_method='wetbulb',
    rain_snow_params={'tw_threshold': 275.15},  # 2°C wet-bulb
    albedo_method='snicar_lite',
    phase_change_method='hierarchy',
    thin_snow_damping=0.7,
    thin_snow_threshold=75.0,
    ground_flux_const=2.0,
)
```

---

## Workflow

### Basic Usage

```python
from pysnow import PySnowModel, SnowConfig

# 1. Configure model
config = SnowConfig(
    rain_snow_method='wetbulb',
    rain_snow_params={'tw_threshold': 275.15},
    albedo_method='snicar_lite',
)

# 2. Initialize model
model = PySnowModel(config)

# 3. Prepare forcing data (n_steps x 8 array)
# Columns: [sw_down, lw_down, precip, t_air, wind_u, wind_v, pressure, q_air]
forcing = np.array([...])

# 4. Run model
outputs = model.run(forcing, dt=3600.0)

# 5. Access results
swe = outputs['swe']           # Snow water equivalent [mm]
depth = outputs['depth']       # Snow depth [m]
melt = outputs['melt']         # Melt rate [mm/timestep]
runoff = outputs['runoff']     # Runoff [mm/timestep]
```

### Forcing Data Requirements

| Variable | Units | Description |
|----------|-------|-------------|
| sw_down | W/m² | Downward shortwave radiation |
| lw_down | W/m² | Downward longwave radiation |
| precip | mm/hr | Precipitation rate |
| t_air | K | Air temperature |
| wind_u | m/s | Eastward wind component |
| wind_v | m/s | Northward wind component |
| pressure | Pa | Surface air pressure |
| q_air | kg/kg | Specific humidity |

### Output Variables

| Variable | Units | Description |
|----------|-------|-------------|
| swe | mm | Snow water equivalent |
| depth | m | Snow depth |
| t_snow | K | Snow temperature |
| albedo | - | Surface albedo |
| density | kg/m³ | Bulk density |
| melt | mm | Melt per timestep |
| runoff | mm | Runoff per timestep |
| snowfall | mm | Snowfall per timestep |
| rainfall | mm | Rainfall per timestep |
| sublimation | mm | Sublimation per timestep (negative = loss) |
| Rn | W/m² | Net radiation |
| H | W/m² | Sensible heat flux |
| LE | W/m² | Latent heat flux |
| G | W/m² | Ground heat flux |

---

## Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Freezing point | $T_{frz}$ | 273.15 | K |
| Water density | $\rho_w$ | 1000 | kg/m³ |
| Ice density | $\rho_i$ | 917 | kg/m³ |
| Ice heat capacity | $c_{p,ice}$ | 2117 | J/(kg·K) |
| Water heat capacity | $c_{p,water}$ | 4186 | J/(kg·K) |
| Air heat capacity | $c_{p,air}$ | 1005 | J/(kg·K) |
| Latent heat of fusion | $L_f$ | 333,500 | J/kg |
| Latent heat of vaporization | $L_v$ | 2,501,000 | J/kg |
| Latent heat of sublimation | $L_s$ | 2,834,500 | J/kg |
| Ice thermal conductivity | $k_{ice}$ | 2.29 | W/(m·K) |
| Air thermal conductivity | $k_{air}$ | 0.023 | W/(m·K) |
| Stefan-Boltzmann constant | $\sigma$ | 5.67×10⁻⁸ | W/(m²·K⁴) |
| Snow emissivity | $\epsilon$ | 0.98 | - |
| Von Karman constant | $\kappa$ | 0.41 | - |
| Gravity | $g$ | 9.81 | m/s² |

---

## References

### Rain-Snow Partitioning
- **Dai, A. (2008)**. Temperature and pressure dependence of the rain-snow phase transition over land and ocean. *Geophysical Research Letters*, 35, L12802. doi:10.1029/2008GL033295

- **Jennings, K. S., Winchell, T. S., Livneh, B., & Molotch, N. P. (2018)**. Spatial variation of the rain-snow temperature threshold across the Northern Hemisphere. *Nature Communications*, 9, 1148. doi:10.1038/s41467-018-03629-7

- **Wang, Y.-H., Broxton, P., Fang, Y., et al. (2019)**. A wet-bulb temperature-based rain-snow partitioning scheme improves snowpack prediction over the drier western United States. *Geophysical Research Letters*, 46, 13825-13835. doi:10.1029/2019GL085722

- **Stull, R. (2011)**. Wet-bulb temperature from relative humidity and air temperature. *Journal of Applied Meteorology and Climatology*, 50, 2267-2269.

### Albedo
- **Essery, R., Morin, S., Lejeune, Y., & Ménard, C. B. (2013)**. A comparison of 1701 snow models using observations from an alpine site. *Advances in Water Resources*, 55, 131-148.

- **Flanner, M. G., & Zender, C. S. (2006)**. Linking snowpack microphysics and albedo evolution. *Journal of Geophysical Research*, 111, D12208.

### Thermal Properties
- **Jordan, R. (1991)**. A one-dimensional temperature model for a snow cover. *U.S. Army Corps of Engineers, Cold Regions Research and Engineering Laboratory*, Special Report 91-16.

- **Sturm, M., Holmgren, J., König, M., & Morris, K. (1997)**. The thermal conductivity of seasonal snow. *Journal of Glaciology*, 43, 26-41.

- **Calonne, N., Flin, F., Morin, S., et al. (2011)**. Numerical and experimental investigations of the effective thermal conductivity of snow. *Geophysical Research Letters*, 38, L23501.

### Compaction
- **Anderson, E. A. (1976)**. A point energy and mass balance model of a snow cover. *NOAA Technical Report NWS 19*.

### Turbulent Fluxes
- **Louis, J.-F. (1979)**. A parametric model of vertical eddy fluxes in the atmosphere. *Boundary-Layer Meteorology*, 17, 187-202.

### Model Evaluation
- **Ryken, A., et al. (2020)**. Sensitivity and model reduction of simulated snow processes. *Advances in Water Resources*, 136, 103512.

---

## Evaluation Results

Comprehensive testing across 8 SNOTEL sites (1995-2024) showed:

| Configuration | Mean |Bias|% | Mean NSE | Best For |
|---------------|-------------|----------|----------|
| Wetbulb-Tw2C | **21.9%** | **0.69** | Overall best |
| Wetbulb-Damp0.7 | 25.6% | 0.64 | Sites with thin snow issues |
| CLM-baseline | 38.5% | 0.48 | Reference comparison |

Key findings:
1. Wet-bulb temperature significantly outperforms air temperature for rain-snow partitioning
2. A 2°C wet-bulb threshold works better than the 1°C default
3. SNICAR-lite albedo performs best among albedo schemes
4. Thin snow damping helps reduce early-season melt oscillations

---

*Documentation generated: January 2025*
