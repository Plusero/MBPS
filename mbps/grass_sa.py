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
tsim = ??? # [d]
dt = ??? # [d]
# Initial conditions
x0 = {???} # [kgC m-2]
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p = {???
     }
# Model parameters adjusted manually to obtain growth
p[???] = ???
p[???] = ???

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], and
# water availability index [-]
t_ini = ???
t_end = ???
data_weather = pd.read_csv(
    '../data/etmgeg_260.csv', # .. to move up one directory from current directory
    skipinitialspace=True, # ignore spaces after comma separator
    header = 47-3, # row with column names, 0-indexed, excluding spaces
    usecols = ['YYYYMMDD', 'TG', 'Q', 'RH'], # columns to use
    index_col = 0, # column with row names from used columns, 0-indexed
    )
# Retrieve relevant arrays from pandas dataframe
T = data_weather.loc[t_ini:t_end,'T'].values    # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end,'Q'].values # [J cm-2 d-1] Global irradiance 
# Aply the necessary conversions of units
T = ???    # [???] to [???] Env. temperature
I0 = ???   # [???] to [???] Global irradiance to PAR
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'T':np.array([tsim, T]).T,
     'I0':np.array([tsim, I0]).T,   
     'WAI':np.array([tsim, np.full((tsim.size,),1.0)]).T
     }

# Controlled inputs
u = {'f_Gr':0, 'f_Hr':0}            # [kgDM m-2 d-1]

# Initialize grass module
grass = Grass(tsim, dt, x0, p)

# Normalized sensitivities
ns = grass.ns(x0, p, d=d, u=u, y_keys=('Wg',))

# Calculate mean NS through time
# TODO: use the ns DataFrame to calculate mean NS per parameter


# -- Plots
# TODO: Make the necessary plots (example provided below)
plt.figure(1)
plt.plot(grass.t, ns['Wg','a','-'], label='a -', linestyle='--')
plt.plot(grass.t, ns['Wg','a','+'], label='a +')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel('normalized sensitivity [-]')
