"""
Physical constants for snow model.
"""

# Temperature
TFRZ = 273.15  # Freezing point [K]

# Densities [kg/m³]
RHO_WATER = 1000.0
RHO_ICE = 917.0
RHO_AIR_SL = 1.225  # Sea level air density

# Heat capacities [J/(kg·K)]
CP_ICE = 2117.0
CP_WATER = 4186.0
CP_AIR = 1005.0

# Latent heats [J/kg]
LF_FUSION = 333500.0  # Latent heat of fusion
LV_VAPORIZATION = 2501000.0  # Latent heat of vaporization at 0°C
LS_SUBLIMATION = LF_FUSION + LV_VAPORIZATION  # Latent heat of sublimation

# Thermal conductivities [W/(m·K)]
TK_WATER = 0.57
TK_ICE = 2.29
TK_AIR = 0.023

# Radiation
STEFAN_BOLTZMANN = 5.67e-8  # [W/(m²·K⁴)]
EMISSIVITY_SNOW = 0.98

# Turbulence
VON_KARMAN = 0.41
GRAVITY = 9.81  # [m/s²]

# Gas constant for dry air
R_AIR = 287.0  # [J/(kg·K)]
