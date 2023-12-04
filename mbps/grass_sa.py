# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- add your team names here --

Sensitivity analysis of the grass growth model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.grass import Grass

plt.style.use('ggplot')

# TODO: Define the required variables for the grass module

# Simulation time
tsim = np.linspace(0.0, 365.0, 365 + 1)  # [d]
dt = 1  # [d]
# Initial conditions
x0 = {'Ws': 0.2, 'Wg': 0.2}  # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
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
p['alpha'] = 2E-9 * 10

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], and
# water availability index [-]
t_ini = '20180101'
t_end = '20190101'
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv',  # .. to move up one directory from current directory
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

# Initialize grass module
grass = Grass(tsim, dt, x0, p)

# Normalized sensitivities
ns = grass.ns(x0, p, d=d, u=u, y_keys=('Wg',))
plot = ns.plot(title="Normalized sensitivities")
# Calculate mean NS through time
# TODO: use the ns DataFrame to calculate mean NS per parameter
mean_NS = ns.mean(axis=0)

mean_NS = mean_NS.abs()
mean_NS_sorted = mean_NS.sort_values(ascending=False)

# mean_NS_sorted = mean_NS.sort_values(by=mean_NS.columns[0], ascending=False)
# print(mean_NS_sorted)

# -- Plots
# TODO: Make the necessary plots (example provided below)


plt.figure(1)
f, [ax1, ax2, ax3, ax4, ax5, ax6, ax7] = plt.subplots(1, 7, sharey='row')  # sharey='row'
ax1.plot(grass.t, ns['Wg', 'phi', '-'], label='phi -', linestyle='--')
ax1.plot(grass.t, ns['Wg', 'phi', '+'], label='phi +')
ax1.legend()
ax2.plot(grass.t, ns['Wg', 'alpha', '-'], label='alpha -', linestyle='--')
ax2.plot(grass.t, ns['Wg', 'alpha', '+'], label='alpha +')
ax2.legend()
ax3.plot(grass.t, ns['Wg', 'beta', '-'], label='beta -', linestyle='--')
ax3.plot(grass.t, ns['Wg', 'beta', '+'], label='beta +')
ax3.legend()
ax4.plot(grass.t, ns['Wg', 'Y', '-'], label='Y -', linestyle='--')
ax4.plot(grass.t, ns['Wg', 'Y', '+'], label='Y +')
ax4.legend()
ax5.plot(grass.t, ns['Wg', 'M', '-'], label='M -', linestyle='--')
ax5.plot(grass.t, ns['Wg', 'M', '+'], label='M +')
ax5.legend()
ax6.plot(grass.t, ns['Wg', 'Tmin', '-'], label='Tmin -', linestyle='--')
ax6.plot(grass.t, ns['Wg', 'Tmin', '+'], label='Tmin +')
ax6.legend()
ax7.plot(grass.t, ns['Wg', 'Topt', '-'], label='Topt -', linestyle='--')
ax7.plot(grass.t, ns['Wg', 'Topt', '+'], label='Topt +')
ax7.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')
plt.show()
#
# plt.plot(grass.t, ns['Wg', 'a', '-'], label='a -', linestyle='--')
# plt.plot(grass.t, ns['Wg', 'a', '+'], label='a +')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')
# plt.subplot(212)
# plt.plot(grass.t, ns['Wg', 'alpha', '-'], label='alpha -', linestyle='--')
# plt.plot(grass.t, ns['Wg', 'alpha', '+'], label='alpha +')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel('normalized sensitivity [-]')
# plt.show()
