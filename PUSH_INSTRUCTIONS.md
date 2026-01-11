# Git Push Instructions

## Summary of Changes

This commit adds new rain-snow partitioning methods and fixes early-season thin snow melt issues.

### Modified Files (core library):
- `pysnow/rain_snow.py` - Added Dai (2008) and Wang et al. (2019) wetbulb_sigmoid methods (+182 lines)
- `pysnow/model.py` - Added thin_snow_damping config parameters (+6 lines)
- `pysnow/thermal.py` - Implemented thin snow damping logic (+15 lines)

### New Files:
- `scripts/run_snotel_site.py` - SNOTEL site model runner with HydroData integration
- `scripts/run_comparison.py` - Multi-configuration comparison script
- `scripts/diagnose_early_season.py` - Early season diagnostic tool
- `scripts/diagnose_energy_balance.py` - Energy balance diagnostic tool
- `docs/SESSION_SUMMARY_2025-01-11.md` - Detailed session documentation

## Commands to Push

```bash
cd /Users/reed/Projects/pysnow_repo

# Review changes
git status
git diff pysnow/rain_snow.py  # Review rain-snow changes
git diff pysnow/thermal.py     # Review thermal changes

# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "Add rain-snow methods (Dai, wetbulb_sigmoid) and thin snow damping fix

New features:
- Dai (2008) hyperbolic tangent rain-snow partitioning
- Wang et al. (2019) wet-bulb sigmoid partitioning
- Thin snow damping to prevent early-season melt-out

New scripts:
- run_snotel_site.py: Automated SNOTEL site comparison
- diagnose_early_season.py: Early season diagnostics
- diagnose_energy_balance.py: Energy balance analysis

Tested against SNOTEL sites:
- CSS Lab CA WY2024: -2% error (Wetbulb method)
- Berthoud Summit CO WY2023: -2% error
- Canyon WY WY2023: -18% error (precip undercatch)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

## Alternative: Create a Feature Branch

If you prefer to review before merging to main:

```bash
cd /Users/reed/Projects/pysnow_repo

# Create feature branch
git checkout -b feature/rain-snow-improvements

# Stage and commit
git add -A
git commit -m "Add rain-snow methods and thin snow damping fix"

# Push feature branch
git push -u origin feature/rain-snow-improvements

# Then create PR on GitHub
```

## Verification After Push

After pushing, verify the changes work:

```bash
cd /Users/reed/Projects/pysnow_repo

# Test imports
python -c "from pysnow.rain_snow import partition_rain_snow; print('OK')"

# Test new methods
python -c "
from pysnow.rain_snow import partition_rain_snow
from pysnow.constants import TFRZ

# Test Dai method
result = partition_rain_snow(1.0, TFRZ + 1.0, method='dai')
print(f'Dai at 1C: {result:.2f}')

# Test wetbulb_sigmoid
result = partition_rain_snow(1.0, TFRZ + 2.0, rh=0.7, pressure=85000, method='wetbulb_sigmoid')
print(f'Wetbulb-sigmoid at 2C, 70% RH: {result:.2f}')
"
```

---
*You can delete this file after pushing.*
