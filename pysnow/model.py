"""
PySnow: Standalone snow model with switchable parameterizations.

Main model driver that combines all physics modules.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .constants import TFRZ, RHO_WATER, CP_ICE
from . import rain_snow
from . import albedo as albedo_module
from . import thermal
from . import compaction
from . import turbulent
from . import radiation


@dataclass
class SnowConfig:
    """Configuration for snow model parameterizations."""

    # Rain-snow partitioning
    rain_snow_method: str = 'linear'
    rain_snow_params: Dict[str, Any] = field(default_factory=dict)

    # Albedo
    albedo_method: str = 'vic'
    albedo_params: Dict[str, Any] = field(default_factory=dict)

    # Thermal conductivity
    conductivity_method: str = 'jordan'

    # Phase change
    phase_change_method: str = 'hierarchy'
    cold_content_tax: bool = False

    # Compaction
    compaction_method: str = 'essery'

    # Turbulent fluxes
    stability_correction: bool = True
    windless_coeff: float = 0.0  # W/(m²·K)
    z_wind: float = 10.0
    z_temp: float = 2.0
    z_0: float = 0.001

    # Ground heat flux
    ground_flux_method: str = 'constant'
    ground_flux_const: float = 2.0  # W/m²
    t_soil: float = TFRZ + 5.0  # Soil temperature [K]

    # Liquid water
    min_water_frac: float = 0.01
    max_water_frac: float = 0.10
    drain_rate: float = 2.778e-5  # m/s (10 cm/hr)


@dataclass
class SnowState:
    """Snow state variables."""
    swe: float = 0.0           # Snow water equivalent [mm]
    depth: float = 0.0         # Snow depth [m]
    density: float = 250.0     # Bulk density [kg/m³]
    t_snow: float = TFRZ       # Snow temperature [K]
    albedo: float = 0.85       # Albedo
    age: float = 0.0           # Snow age [seconds]
    liquid_water: float = 0.0  # Liquid water content [mm]
    grain_radius: float = 50.0 # Grain radius [μm] (for SNICAR)


@dataclass
class SnowFluxes:
    """Output fluxes for a timestep."""
    precip: float = 0.0        # Total precipitation [mm]
    snowfall: float = 0.0      # Snowfall [mm]
    rainfall: float = 0.0      # Rainfall [mm]
    melt: float = 0.0          # Melt [mm]
    refreeze: float = 0.0      # Refreeze [mm]
    sublimation: float = 0.0   # Sublimation [mm] (negative = loss)
    runoff: float = 0.0        # Runoff [mm]
    Rn: float = 0.0            # Net radiation [W/m²]
    H: float = 0.0             # Sensible heat [W/m²]
    LE: float = 0.0            # Latent heat [W/m²]
    G: float = 0.0             # Ground heat [W/m²]


class PySnowModel:
    """
    Standalone snow model with modular physics.

    Examples
    --------
    >>> config = SnowConfig(rain_snow_method='jennings', albedo_method='vic')
    >>> model = PySnowModel(config)
    >>> state = model.step(forcing, dt=3600)
    """

    def __init__(self, config: Optional[SnowConfig] = None):
        """
        Initialize snow model.

        Parameters
        ----------
        config : SnowConfig, optional
            Model configuration. Uses defaults if not provided.
        """
        self.config = config if config else SnowConfig()
        self.state = SnowState()
        self.fluxes = SnowFluxes()

    def reset(self):
        """Reset model state to initial conditions."""
        self.state = SnowState()
        self.fluxes = SnowFluxes()

    def step(self, forcing: Dict[str, float], dt: float = 3600.0) -> SnowState:
        """
        Advance model by one timestep.

        Parameters
        ----------
        forcing : dict
            Forcing data with keys:
            - t_air: Air temperature [K]
            - precip: Precipitation rate [mm/hr]
            - sw_down: Shortwave radiation [W/m²]
            - lw_down: Longwave radiation [W/m²]
            - wind: Wind speed [m/s]
            - pressure: Air pressure [Pa]
            - q_air: Specific humidity [kg/kg]
        dt : float
            Timestep in seconds

        Returns
        -------
        SnowState
            Updated snow state
        """
        cfg = self.config
        state = self.state
        fluxes = SnowFluxes()

        # Unpack forcing
        t_air = forcing['t_air']
        # precip input is in mm/hr, convert to mm for this timestep
        precip = forcing['precip'] * dt / 3600.0
        sw_down = forcing['sw_down']
        lw_down = forcing['lw_down']
        wind = max(0.1, forcing['wind'])  # Minimum wind speed
        pressure = forcing['pressure']
        q_air = forcing['q_air']

        # Calculate relative humidity
        rh = rain_snow.calc_rh_from_specific_humidity(q_air, t_air, pressure)

        fluxes.precip = precip

        # =====================================================================
        # 1. Rain-Snow Partitioning
        # =====================================================================
        if precip > 0:
            snow_frac = rain_snow.partition_rain_snow(
                precip, t_air, rh=rh, pressure=pressure,
                method=cfg.rain_snow_method, **cfg.rain_snow_params
            )
            fluxes.snowfall = precip * snow_frac
            fluxes.rainfall = precip * (1 - snow_frac)
        else:
            fluxes.snowfall = 0
            fluxes.rainfall = 0

        # =====================================================================
        # 2. Add New Snow
        # =====================================================================
        if fluxes.snowfall > 0:
            # Fresh snow density
            rho_new = compaction.calc_new_snow_density(t_air)

            # Add to pack
            state.density, state.swe, state.depth = compaction.add_fresh_snow(
                state.density, state.swe, state.depth,
                fluxes.snowfall, rho_new
            )

            # Reset snow age (weighted by new snow fraction)
            if state.swe > 0:
                frac_new = fluxes.snowfall / state.swe
                state.age = state.age * (1 - frac_new)

            # Reset grain radius for SNICAR
            if state.swe > 0:
                frac_new = fluxes.snowfall / state.swe
                state.grain_radius = frac_new * 50.0 + (1 - frac_new) * state.grain_radius

        # Add rainfall to liquid water
        if fluxes.rainfall > 0 and state.swe > 0:
            state.liquid_water += fluxes.rainfall

        # =====================================================================
        # 3. Snow Physics (only if snow exists)
        # =====================================================================
        if state.swe > 0:
            # Age snow
            state.age += dt

            # Compaction
            state.density, state.depth = compaction.update_density(
                state.density, state.swe, state.depth, state.t_snow, dt,
                method=cfg.compaction_method
            )

            # Albedo
            albedo_result = albedo_module.calc_albedo(
                method=cfg.albedo_method,
                age=state.age,
                t_snow=state.t_snow,
                snowfall=fluxes.snowfall,
                swe=state.swe,
                dt=dt,
                albedo_prev=state.albedo,
                grain_radius=state.grain_radius,
                **cfg.albedo_params
            )
            state.albedo = albedo_result[0]
            if albedo_result[1] is not None:
                state.grain_radius = albedo_result[1]

            # Thin snow correction
            state.albedo = albedo_module.thin_snow_correction(
                state.albedo, state.depth
            )

            # -----------------------------------------------------------------
            # Energy Balance
            # -----------------------------------------------------------------

            # Radiation
            rad = radiation.calc_net_radiation(
                sw_down, lw_down, state.albedo, state.t_snow
            )
            fluxes.Rn = rad['Rn']

            # Turbulent fluxes
            turb = turbulent.calc_turbulent_fluxes(
                t_air, state.t_snow, wind, rh, pressure,
                z_wind=cfg.z_wind, z_temp=cfg.z_temp, z_0=cfg.z_0,
                stability_correction=cfg.stability_correction,
                windless_coeff=cfg.windless_coeff
            )
            fluxes.H = turb['H']
            fluxes.LE = turb['LE']

            # Ground heat flux
            thk_snow = thermal.calc_thermal_conductivity(
                state.density, method=cfg.conductivity_method
            )
            fluxes.G = turbulent.calc_ground_heat_flux(
                state.t_snow, cfg.t_soil, state.depth, thk_snow,
                method=cfg.ground_flux_method, G_const=cfg.ground_flux_const
            )

            # Net energy [W/m²]
            Q_net = fluxes.Rn + fluxes.H + fluxes.LE + fluxes.G

            # Convert to J/m² for timestep
            energy = Q_net * dt

            # -----------------------------------------------------------------
            # Phase Change
            # -----------------------------------------------------------------
            phase_result = thermal.apply_energy_to_snowpack(
                energy, state.swe, state.t_snow, state.liquid_water,
                method=cfg.phase_change_method,
                cold_content_tax=cfg.cold_content_tax
            )

            state.swe = phase_result['swe']
            state.t_snow = phase_result['t_snow']
            state.liquid_water = phase_result['liquid_water']
            fluxes.melt = phase_result['melt']
            fluxes.refreeze = phase_result['refreeze']

            # Update depth
            # depth [m] = SWE [kg/m²] / density [kg/m³]
            # Note: SWE in mm = kg/m² (since 1mm water = 1 kg/m²)
            if state.swe > 0 and state.density > 0:
                ice = state.swe - state.liquid_water
                state.depth = ice / state.density

            # -----------------------------------------------------------------
            # Sublimation
            # -----------------------------------------------------------------
            if turb['E_mass'] < 0:  # Mass loss (sublimation)
                sublim = abs(turb['E_mass']) * dt  # kg/m² = mm
                sublim = min(sublim, state.swe)
                state.swe -= sublim
                fluxes.sublimation = -sublim

                # Remove proportionally from ice and liquid
                if state.swe > 0:
                    state.liquid_water = state.liquid_water * (state.swe / (state.swe + sublim))
                else:
                    state.liquid_water = 0

            # -----------------------------------------------------------------
            # Liquid Water Drainage
            # -----------------------------------------------------------------
            state.liquid_water, fluxes.runoff = thermal.update_liquid_water(
                state.liquid_water, state.depth, dt,
                min_frac=cfg.min_water_frac, max_frac=cfg.max_water_frac,
                drain_rate=cfg.drain_rate
            )

            # CRITICAL: Reduce SWE by runoff amount
            # Runoff is water leaving the snowpack
            state.swe -= fluxes.runoff
            state.swe = max(0, state.swe)

        else:
            # No snow - reset state
            state.swe = 0
            state.depth = 0
            state.density = 250.0
            state.t_snow = min(t_air, TFRZ)
            state.albedo = 0.85
            state.age = 0
            state.liquid_water = 0
            state.grain_radius = 50.0

        # Safety: ensure temperature is physical
        state.t_snow = max(173.15, min(TFRZ, state.t_snow))

        # Store fluxes
        self.fluxes = fluxes

        return state

    def run(self, forcing_data: np.ndarray, dt: float = 3600.0,
            verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Run model for multiple timesteps.

        Parameters
        ----------
        forcing_data : np.ndarray
            Array of shape (n_steps, 8) with columns:
            [sw_down, lw_down, precip, t_air, wind_u, wind_v, pressure, q_air]
        dt : float
            Timestep in seconds
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Output arrays with keys: swe, depth, t_snow, albedo, melt, runoff, etc.
        """
        n_steps = len(forcing_data)

        # Output arrays
        outputs = {
            'swe': np.zeros(n_steps),
            'depth': np.zeros(n_steps),
            't_snow': np.zeros(n_steps),
            'albedo': np.zeros(n_steps),
            'density': np.zeros(n_steps),
            'melt': np.zeros(n_steps),
            'runoff': np.zeros(n_steps),
            'snowfall': np.zeros(n_steps),
            'rainfall': np.zeros(n_steps),
            'sublimation': np.zeros(n_steps),
            'Rn': np.zeros(n_steps),
            'H': np.zeros(n_steps),
            'LE': np.zeros(n_steps),
            'G': np.zeros(n_steps),
        }

        self.reset()

        for i in range(n_steps):
            if verbose and i % 1000 == 0:
                print(f"Step {i}/{n_steps}")

            # Unpack forcing
            forcing = {
                'sw_down': forcing_data[i, 0],
                'lw_down': forcing_data[i, 1],
                'precip': forcing_data[i, 2],  # mm/hr
                't_air': forcing_data[i, 3],
                'wind': np.sqrt(forcing_data[i, 4]**2 + forcing_data[i, 5]**2),
                'pressure': forcing_data[i, 6],
                'q_air': forcing_data[i, 7],
            }

            # Step model
            state = self.step(forcing, dt)

            # Store outputs
            outputs['swe'][i] = state.swe
            outputs['depth'][i] = state.depth
            outputs['t_snow'][i] = state.t_snow
            outputs['albedo'][i] = state.albedo
            outputs['density'][i] = state.density
            outputs['melt'][i] = self.fluxes.melt
            outputs['runoff'][i] = self.fluxes.runoff
            outputs['snowfall'][i] = self.fluxes.snowfall
            outputs['rainfall'][i] = self.fluxes.rainfall
            outputs['sublimation'][i] = self.fluxes.sublimation
            outputs['Rn'][i] = self.fluxes.Rn
            outputs['H'][i] = self.fluxes.H
            outputs['LE'][i] = self.fluxes.LE
            outputs['G'][i] = self.fluxes.G

        return outputs
