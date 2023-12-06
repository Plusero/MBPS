# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Evaluation of the grass & water model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass_sol import Grass
from mbps.models.water_sol import Water

verification_date = [114,
                     129,
                     140,
                     149,
                     163]
verification = [40.3932797998184,
                135.05780009766931,
                259.5707117634952,
                552.0907317416204,
                691.1513254337085]
VERIVICATION_date = [115,
                     130,
                     141,
                     150,
                     164,
                     176,
                     186]
# verification = [40.3932797998184,
#                 135.05780009766931,
#                 259.5707117634952,
#                 552.0907317416204,
#                 691.1513254337085]
VERIVICATION = [38.46364823579006,
                133.4911616122081,
                257.8781809495299,
                550.571525667127,
                689.5948766114153,
                1254.6804969101136,
                1279.7563315457844]
plt.style.use('ggplot')

# Simulation time
tsim = np.linspace(0, 365, int(365 / 5) + 1)  # [d]

# Weather data (disturbances shared across models)
# t_ini = '19950101'
# t_end = '19960101'
t_ini = '1'
t_end = '365'
t_weather = np.linspace(0, 364, 365)
data_weather = pd.read_csv(
    '../data/NASA_DATA.csv',  # .. to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    usecols=['DoY', 'Rain', 'Temperature', 'allsky_irri_J'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)
# header=47 - 3,  # row with column names, 0-indexed, excluding spaces
# Grass data. (Organic matter assumed equal to DM) [gDM m-2]
# Groot and Lantinga (2004)
# t_data = np.array(VERIVICATION_date)
# m_data = np.array(VERIVICATION)
t_data = np.array(verification_date)
m_data = np.array(verification)
m_data = m_data / 1E3  # convert gDm to KgDM
# This data was gathered in Wageningen 1995.

# ---- Grass sub-model
# Step size
dt_grs = 1  # [d]x

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 0, 'Wg': 0.02}  # [kgC m-2] according to the Mohtar et al. 1997
# note that the Unit here is KgC m-2, meaning the weight of carbon content
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
         'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
         'beta': 0.05,  # [d-1] senescence rate
         'k': 0.5,  # [-] extinction coefficient of canopy
         'm': 0.1,  # [-] leaf transmission coefficient
         'M': 0.02,  # [d-1] maintenance respiration coefficient
         'mu_m': 0.5,  # [d-1] max. structural specific growth rate
         'P0': 0.432,  # [kgCO2 m-2 d-1] max photosynthesis parameter
         'phi': 0.9,  # [-] photoshynth. fraction for growth
         'Tmin': 0.0,  # [°C] maximum temperature for growth
         'Topt': 20.0,  # [°C] minimum temperature for growth
         'Tmax': 42.0,  # [°C] optimum temperature for growth
         'Y': 0.75,  # [-] structure fraction from storage
         'z': 1.33  # [-] bell function power
         }
# Model parameters adjusted manually to obtain growth
# TODO: Adjust a few parameters to obtain growth.
# Start by using the modifications from Case 1.
# If needed, adjust further those or additional parameters
p_grs['alpha'] = 2E-9 * 10
p_grs['a'] = 21.35
# p_grs['beta'] = 7.724E-2

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'Temperature'].values  # [1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'allsky_irri_J'].values  # [J m-2 d-1] Global irr.

T = T / 1  # [1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl / dt_grs  # [J m-2 d-1] to [J m-2 d-1] Global irr. to PAR

d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

# ---- Water sub-model
dt_wtr = 1  # [d]

# Initial conditions
# TODO: Specify suitable initial conditions for the soil water sub-model
x0_wtr = {'L1': 40, 'L2': 60, 'L3': 100, 'DSD': 2}  # 3*[mm], [d]
# using the values from the Castellaro et al. 2009

# Castellaro et al. 2009, and assumed values for soil types and layers
p_wtr = {'S': 10,  # [mm d-1] parameter of precipitation retention
         'alpha': 1.29E-6,  # [mm J-1] Priestley-Taylor parameter
         'gamma': 0.68,  # [mbar °C-1] Psychrometric constant
         'alb': 0.23,  # [-] Albedo (assumed constant crop & soil)
         'kcrop': 0.90,  # [mm d-1] Evapotransp coefficient, range (0.85-1.0)
         'WAIc': 0.75,  # [-] WDI critical, range (0.5-0.8)
         'theta_fc1': 0.36,  # [-] Field capacity of soil layer 1
         'theta_fc2': 0.32,  # [-] Field capacity of soil layer 2
         'theta_fc3': 0.24,  # [-] Field capacity of soil layer 3
         'theta_pwp1': 0.21,  # [-] Permanent wilting point of soil layer 1
         'theta_pwp2': 0.17,  # [-] Permanent wilting point of soil layer 2
         'theta_pwp3': 0.10,  # [-] Permanent wilting point of soil layer 3
         'D1': 150,  # [mm] Depth of Soil layer 1
         'D2': 250,  # [mm] Depth of soil layer 2
         'D3': 600,  # [mm] Depth of soil layer 3
         'krf1': 0.25,  # [-] Rootfraction layer 1 (guess)
         'krf2': 0.50,  # [-] Rootfraction layer 2 (guess)
         'krf3': 0.25,  # [-] Rootfraction layer 2 (guess)
         'mlc': 0.2,  # [-] Fraction of soil covered by mulching
         }
p_wtr['krop'] = 0.85
# Disturbances
# global irradiance [J m-2 d-1], environment temperature [°C],
# precipitation [mm d-1], leaf area index [-].
T = data_weather.loc[t_ini:t_end, 'Temperature'].values  # [1 °C] Env. temperature
I_glb = data_weather.loc[t_ini:t_end, 'allsky_irri_J'].values  # [J cm-2 d-1] Global irr.
f_prc = data_weather.loc[t_ini:t_end, 'Rain'].values  # [1 mm d-1] Precipitation
f_prc[f_prc < 0.0] = 0  # correct data that contains -0.1 for very low values

T = T
I_glb = I_glb / dt_wtr  # [J m-2 d-1] to [J m-2 d-1] Global irradiance
f_prc = f_prc / dt_wtr  # [1 mm d-1] to [mm d-1] Precipitation

d_wtr = {'I_glb': np.array([t_weather, I_glb]).T,
         'T': np.array([t_weather, T]).T,
         'f_prc': np.array([t_weather, f_prc]).T,
         }

# Initialize module
water = Water(tsim, dt_wtr, x0_wtr, p_wtr)

# ---- Run simulation
# Initial disturbance
d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T

# Iterator
# (stop at second-to-last element, and store index in Fortran order)
it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx + 1])
    print('Integrating', tspan)
    # Controlled inputs
    u_grs = {'f_Gr': 0,  # 0.3 * 15 * 1e-4 * 0.4
             'f_Hr': 0}
    # f_Gr [kgDM m-2 d-1]  Graze dry matter,
    # 3 cows per one ha,value from farmer's knowledge, 15 [kgDM d-1] from Mohtar et al. 1997
    # Harvest dry matter
    u_wtr = {'f_Irg': 0}  # [mm d-1] Irrigation
    # Run grass model
    y_grs = grass.run(tspan, d_grs, u_grs)
    # Retrieve grass model outputs for water model
    d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
    # Run water model
    y_wtr = water.run(tspan, d_wtr, u_wtr)
    # Retrieve water model outputs for grass model
    d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

# Retrieve simulation results
t_grs, t_wtr = grass.t, water.t
WsDM, WgDM, LAI = grass.y['Ws'] / 0.4, grass.y['Wg'] / 0.4, grass.y['LAI']
L1, L2, L3 = water.y['L1'], water.y['L2'], water.y['L3'],
WAI = water.y['WAI']

# calculate Root mean square error
RMSE = np.sqrt(np.mean((m_data - WgDM[t_data]) ** 2))
print('RMSE = ', RMSE)
# ---- Plots
plt.figure(1)
plt.plot(t_grs, WsDM, label='WsDM')
plt.plot(t_grs, WgDM, label='WgDM')
plt.plot(t_data, m_data,
         linestyle='None', marker='o', label='WgDM data')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$grass\ biomass\ [kgDM\ m^{-2}]$')
plt.show()

# References
# Groot, J.C.J., and Lantinga, E.A., (2004). An object oriented model
#   of the morphological development and digestability of perennial
#   ryegrass. Ecological Modelling 177(3-4), 297-312.
