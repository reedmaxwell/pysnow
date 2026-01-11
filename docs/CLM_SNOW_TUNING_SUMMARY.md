# CLM Snow Model Tuning and PySnow Development Summary

## Background

This document summarizes the investigation into CLM snow model biases in ParFlow and the development of PySnow, a standalone Python snow model for rapid parameterization testing.

**Test Site:** CSS Lab SNOTEL (428:CA:SNTL), California Sierra Nevada
**Period:** Water Year 2023-2024 (October 1, 2023 - October 1, 2024)
**Observations:** SNOTEL SWE measurements (peak: 1024mm on April 2, melt-out: ~May 21)

---

## Problem Statement

The CLM snow model in ParFlow exhibited two primary biases:

1. **Lower peak SWE than observations** (~8-20% underestimate)
2. **Early-season "sawtooth" pattern** - snow accumulates then melts back between events
3. **Late-season slow melt** - model melts out 2-4 weeks after observations

---

## ParFlow-CLM Tuning Rounds

### Round 1: Initial Fixes
| Parameter | Original | Changed To | Purpose |
|-----------|----------|------------|---------|
| Melt threshold | `tfrz` | `tfrz + 0.5` | Prevent spurious melt |
| Albedo decay (VIS) | 0.5 | 0.2 | Reduce aging |
| Albedo decay (NIR) | 0.2 | 0.05 | Reduce aging |
| Snow aging rate | 1.0e-6 | 5.0e-7 | Slower aging |

**Result:** Peak SWE improved from 936mm to 1017mm (+8.6%)

### Round 2: Rain-Snow Threshold
| Parameter | Original | Changed To | Purpose |
|-----------|----------|------------|---------|
| 100% snow threshold | 0°C | +1°C | More precip as snow |
| Transition end | +2°C | +3°C | Wider snow zone |

**Result:** Peak SWE 1026mm (+0.3% vs SNOTEL), but melt-out still ~16 days late

### Round 3-9: Various Experiments
- Albedo adjustments
- Thermal conductivity (+50%)
- Initial temperature (300K → 282K)
- Jennings humidity-dependent rain-snow model

**Key Finding:** Late-season slow melt persisted across all configurations. The Jennings model actually reduced peak SWE because high RH during precipitation caused more rain classification.

---

## PySnow Development

### Motivation

To rapidly test combinations of snow physics parameterizations without recompiling ParFlow, we extracted snow model physics from CLM, CoLM, and SNOWCLIM into a standalone Python package.

### Architecture

```
pysnow/
├── __init__.py          # Package initialization
├── constants.py         # Physical constants (TFRZ, LF_FUSION, etc.)
├── rain_snow.py         # Rain-snow partitioning schemes
├── albedo.py            # Albedo calculation methods
├── thermal.py           # Thermal properties and phase change
├── compaction.py        # Density evolution
├── turbulent.py         # Sensible/latent heat fluxes
├── radiation.py         # Net radiation balance
├── model.py             # Main driver (PySnowModel class)
└── run_comparison.py    # Test script
```

### Available Parameterizations

**Rain-Snow Partitioning:**
- `threshold`: Simple temperature threshold
- `linear`: Linear ramp between thresholds (CLM default)
- `jennings`: Humidity-dependent logistic (Jennings et al. 2018)
- `wet_bulb`: Wet-bulb temperature threshold (CoLM)

**Albedo:**
- `constant`: Fixed value
- `clm_decay`: Exponential decay with age
- `vic`: Separate cold/thaw decay coefficients
- `essery`: Essery et al. (2013) with snow refresh
- `tarboton`: UEB temperature-dependent aging
- `snicar_lite`: Simplified SNICAR with grain size evolution

**Phase Change:**
- `hierarchy`: SNOWCLIM-style (cold content → refreeze → melt)
- `clm`: CLM-style simultaneous

**Compaction:**
- `essery`: Viscosity-based (Essery et al. 2013)
- `anderson`: Anderson (1976) - CLM style
- `simple`: Time-based

---

## Bugs Discovered and Fixed

### 1. Depth Unit Conversion (Critical)
**Bug:** `depth = swe / density / 1000.0` made depth 1000x too small
**Fix:** `depth = swe / density` (SWE in mm = kg/m², density in kg/m³)
**Impact:** Caused thin snow albedo correction to kick in for all snow < 100m depth

### 2. Thin Snow Thermal Instability
**Bug:** With 1mm SWE, heat capacity is ~2000 J/K, so 60 W/m² causes 100°C temperature swings
**Fix:** Added minimum effective mass (5mm) for energy balance calculations
**Impact:** Prevented unrealistic temperature oscillations (-100°C to 0°C)

### 3. Missing Runoff-SWE Coupling
**Bug:** Liquid water drainage (runoff) didn't reduce total SWE
**Fix:** `state.swe -= fluxes.runoff` after drainage calculation
**Impact:** Snow accumulated but never melted out

### 4. Thin Snow Albedo Too Aggressive
**Bug:** Critical depth for albedo blending was 10cm with exponential formula
**Fix:** Reduced to 2cm with linear blending
**Impact:** Even 1cm of snow now maintains reasonable albedo

---

## PySnow Results vs SNOTEL

| Configuration | Peak SWE | vs SNOTEL | Melt-out | vs SNOTEL |
|--------------|----------|-----------|----------|-----------|
| **SNOTEL** | 1024 mm | - | May 21 | - |
| SNICAR-lite | 897 mm | -12% | May 13 | -8 days |
| CLM-like | 819 mm | -20% | June 4 | +14 days |
| SNOWCLIM-hierarchy | 799 mm | -22% | **May 20** | **-1 day** |
| VIC-albedo | 717 mm | -30% | May 8 | -13 days |
| Combined | 522 mm | -49% | Apr 23 | -28 days |

**Best configurations:**
- **Peak SWE:** SNICAR-lite (-12%)
- **Melt timing:** SNOWCLIM-hierarchy (-1 day)

---

## Key Physical Findings

### 1. Early-Season Sawtooth Pattern
The sawtooth pattern (snow accumulates then melts back) is a **physics problem**, not an implementation bug. It appears in both ParFlow-CLM and PySnow with identical forcing.

**Likely causes:**
- Ground heat flux (2 W/m²) melts thin snow from below
- Warm initial soil temperature
- Thin snow has low heat capacity, responds quickly to energy inputs

### 2. Melt Timing Sensitivity
- **CLM-style phase change** produces late melt (conservative)
- **SNOWCLIM hierarchy** produces accurate melt timing
- **VIC albedo** with thaw coefficients accelerates spring melt

### 3. Rain-Snow Partitioning
- 75% of precipitation falls as snow with linear method (+1°C to +3°C thresholds)
- Jennings humidity method reduces snowfall because RH is high during precipitation events
- This site may need adjusted thresholds due to elevation/climate

---

## Annual Water Balance (SNOWCLIM-hierarchy)

| Component | Amount | % of Precip |
|-----------|--------|-------------|
| Total precipitation | 1545 mm | 100% |
| Snowfall | 1164 mm | 75% |
| Rainfall | 380 mm | 25% |
| Peak SWE | 799 mm | 52% |
| Total melt | 1331 mm | - |
| Total runoff | 1100 mm | - |
| Sublimation | 79 mm | 5% |

**Snow retention efficiency:** 799 / 1164 = 69% (31% lost before peak)

---

## Recommendations

### For ParFlow-CLM
1. Reduce ground heat flux for thin snowpacks
2. Consider SNOWCLIM-style energy hierarchy for melt timing
3. Evaluate site-specific rain-snow thresholds

### For Future PySnow Development
1. Add multi-layer snow model option
2. Implement wind redistribution
3. Add vegetation interception
4. Couple to simple land surface model

---

## Files and Locations

**PySnow Package:** `/Users/reed/Projects/ParFlow_example_cases/single_column_SWE/pysnow/`

**Forcing Data:** `forcing1D.Css_Lab.2023-10-01-2024-10-01.temp_adjusted.txt`

**Comparison Plot:** `sensitivity_study/pysnow_comparison.png`

---

## References

- Anderson, E.A. (1976). A point energy and mass balance model of a snow cover. NOAA Technical Report NWS 19.
- Essery, R., et al. (2013). A comparison of four different methods of snow model calibration. Hydrological Processes.
- Jennings, K.S., et al. (2018). Spatial variation of the rain-snow temperature threshold across the Northern Hemisphere. Nature Communications.
- Oleson, K.W., et al. (2013). Technical description of version 4.5 of the Community Land Model (CLM). NCAR Technical Note.
