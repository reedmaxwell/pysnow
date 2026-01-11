"""
PySnow: Standalone snow model with switchable parameterizations.

A modular snow model combining physics from CLM, CoLM, and SNOWCLIM.
Designed for rapid testing of parameterization combinations.

Usage
-----
>>> from pysnow import PySnowModel, SnowConfig
>>>
>>> # Configure model
>>> config = SnowConfig(
...     rain_snow_method='linear',      # 'threshold', 'linear', 'jennings', 'wet_bulb'
...     albedo_method='vic',            # 'constant', 'clm_decay', 'vic', 'essery', 'snicar_lite'
...     phase_change_method='hierarchy' # 'hierarchy' (SNOWCLIM), 'clm'
... )
>>>
>>> # Create and run model
>>> model = PySnowModel(config)
>>> outputs = model.run(forcing_data, dt=3600)

Available Methods
-----------------
Rain-Snow Partitioning:
    - 'threshold': Simple temperature threshold
    - 'linear': Linear ramp between thresholds (CLM default)
    - 'jennings': Humidity-dependent logistic (Jennings et al. 2018)
    - 'wet_bulb': Wet-bulb temperature threshold (CoLM option)

Albedo:
    - 'constant': Fixed albedo value
    - 'clm_decay': Simple exponential decay with age
    - 'vic': VIC model with separate cold/thaw coefficients
    - 'essery': Essery et al. (2013) with new snow refresh
    - 'tarboton': UEB model temperature-dependent aging
    - 'snicar_lite': Simplified SNICAR with grain size evolution

Thermal Conductivity:
    - 'jordan': Jordan (1991) - CLM/CoLM default
    - 'sturm': Sturm et al. (1997)
    - 'calonne': Calonne et al. (2011)

Phase Change:
    - 'hierarchy': SNOWCLIM-style (CC → refreeze → melt)
    - 'clm': CLM-style (simultaneous)

Compaction:
    - 'essery': Essery et al. (2013) with viscosity
    - 'anderson': Anderson (1976) - CLM style
    - 'simple': Simple time-based compaction
"""

from .model import PySnowModel, SnowConfig, SnowState, SnowFluxes
from .constants import TFRZ, RHO_WATER, RHO_ICE

__version__ = '0.1.0'
__all__ = ['PySnowModel', 'SnowConfig', 'SnowState', 'SnowFluxes', 'TFRZ']
