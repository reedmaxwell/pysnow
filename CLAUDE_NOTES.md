# Notes for Claude Code Sessions

## Conda Environment

**ALWAYS use the `subsettools` conda environment when running Python scripts in this project.**

```bash
source /Users/reed/miniforge3.1/etc/profile.d/conda.sh && conda activate subsettools && python <script.py>
```

The `parflow` environment does NOT exist. Do not try to use it.

## Key Dependencies in subsettools env
- numpy, pandas, matplotlib
- hf_hydrodata (HydroData API)
- subsettools
- pysnow (from /Users/reed/Projects/pysnow_repo)

## HydroData API Patterns

Correct way to get SNOTEL metadata:
```python
import hf_hydrodata as hf
sites = hf.get_point_metadata(
    dataset='snotel',
    variable='swe',
    site_ids=[site_id]
)
lat = sites.iloc[0]['latitude']
lon = sites.iloc[0]['longitude']
```

Correct way to get SNOTEL data:
```python
swe_data = hf.get_point_data(
    dataset='snotel',
    variable='swe',
    temporal_resolution='daily',
    site_ids=[site_id],
    start_date=start_date,
    end_date=end_date
)
```

## Common Mistakes to Avoid
1. Using wrong conda env (parflow instead of subsettools)
2. Using `hf.get_point()` instead of `hf.get_point_metadata()`
3. Creating new scripts from scratch instead of reusing working code patterns
