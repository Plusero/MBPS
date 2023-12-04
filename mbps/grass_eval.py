# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
M.Sc. Biosystems Engineering, WUR
@authors:   -- add your team names here --

Evaluation of the grass growth model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass import Grass

plt.style.use('ggplot')

# Grass data
# TODO: define numpy arrays with measured grass data in the Netherlands
t_grass_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_grass_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_grass_data = m_grass_data / 1E3  # from [gDM m-2] [kgDM m-2]
# Simulation time
tsim = np.linspace(0.0, 365.0, 365 + 1)  # [d]
dt = 1  # [d]

# Initial conditions
# TODO: define sensible values for the initial conditions
x0 = {'Ws': 0.2, 'Wg': 0.2}  # [kgC m-2]

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the varameper values in the dictionary p
p = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
     'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
     'beta': 0.05,  # 0.05[d-1] senescence rate
     'k': 0.5,  # [-] extinction coefficient of canopy
     'm': 0.1,  # [-] leaf transmission coefficient
     'M': 0.02,  # [d-1] maintenance respiration coefficient
     'mu_m': 0.5,  # [d-1] max. structural specific growth rate
     'P0': 0.432,  # [kgCO2 m-2 d-1] max photosynthesis parameter
     'phi': 0.9,  # [-] photoshynthetic fraction for growth
     'Tmax': 42,  # [°C] maximum temperature for growth
     'Tmin': 0,  # [°C] minimum temperature for growth
     'Topt': 20,  # [°C] optimum temperature for growth
     'Y': 0.75,  # [-] structure fraction from storage
     'z': 1.33  # [-] bell function power
     }

# Model parameters adjusted manually to obtain growth
# TODO: (Further) adjust 2-3 parameter values to match measured growth behaviour
p['alpha'] = 2E-9 * 10

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and water availability index [-]
# TODO: Specify the corresponding dates to read weather data (see csv file).
t_ini = '20180101'
t_end = '20190101'
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv',  # … to move up one directory from current directory
    skipinitialspace=True,  # ignore spaces after comma separator
    header=47 - 3,  # row with column names, 0-indexed, excluding spaces
    usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
    index_col=0,  # column with row names from used columns, 0-indexed
)
# Retrieve relevant arrays from pandas dataframe
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irradiance
# Aply the necessary conversions of units
T = T / 10  # [0.1°C] to [°C] Env. temperature
I0 = I_gl * 1e4  # [J cm-2 d-1] to [J m-2 d-1] Global irradiance to PAR
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'T': np.array([tsim, T]).T,
     'I0': np.array([tsim, I0]).T,
     'WAI': np.array([tsim, np.full((tsim.size,), 1.0)]).T
     }

# Controlled inputs
u = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]

# Initialize module
# TODO: Call the module Grass to initialize an instance
grass = Grass(tsim, dt, x0, p)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y_grass = grass.run(tspan, d, u)

# Retrieve simulation results
# assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
# TODO: Retrieve the simulation results
t_grass = y_grass['t']
WsDM = y_grass['Ws'] * 0.4
WgDM = y_grass['Wg'] * 0.4
W = WsDM + WgDM
# Plot
# TODO: Make a plot for WsDM, WgDM and grass measurement data.
plt.figure(1)
plt.plot(t_grass_data, m_grass_data, label='measurement data')
plt.plot(t_grass, WsDM, label='WsDM')
plt.plot(t_grass, WgDM, label='WgDM')
plt.plot(t_grass, W, label='W')

plt.legend()
plt.figure(2)
plt.plot(t_grass, T, label='T')
plt.xlabel('time' + r'$[d]$')
plt.ylabel('temperature' + r'$^\circ C$' + r'$[-]$')
# plt.legend()
plt.figure(3)
plt.plot(t_grass_data, m_grass_data, "--o")
plt.xlim([0, 365])
plt.xlabel('time' + r'$[d]$')
plt.ylabel('grass biomass' + r'$[kg\cdot m^{-2}]$')
plt.show()
