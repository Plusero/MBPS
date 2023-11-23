# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names --

Initial test of the soil water model
"""

import numpy as np
import matplotlib.pyplot as plt

from mbps.models.water import Water

plt.style.use('ggplot')

# Simulation time
# TODO: Define the simulation time array and integration time step
tsim = np.linspace(0.0, 365.0, 365 + 1)
dt = 1

# Initial conditions
# TODO: define the dictionary of initial conditions
x0 = {'L1': 1, 'L2': 1, 'L3': 1, 'DSD': 2}
# Castellaro et al. 2009, and assumed values for soil types and layers
# Define the dictonary of parameter values
p = {'kcrop': 0.90,
     'gamma': 0.68,
     'alpha': 1.29 * 1e-6,
     'alb': 0.23,
     'WAIc': 0.75,
     'theta_fc1': 0.36,
     'theta_fc2': 0.32,
     'theta_fc3': 0.24,
     'theta_pwp1': 0.21,
     'theta_pwp2': 0.17,
     'theta_pwp3': 0.10,
     'D1': 150,
     'D2': 250,
     'D3': 600,
     'krf1': 0.25,
     'krf2': 0.5,
     'krf3': 0.25,
     'mlc': 0.2,
     'S': 10,
     }

# Disturbances (assumed constant for test)
# global irradiance [J m-2 d-1], environment temperature [°C], 
# precipitation [mm d-1], leaf area index [-]
# TODO: Define sensible constant values for the disturbances
d = {'I_glb': np.array([tsim, np.full((tsim.size,), 86400 * 100 * 5)]).T,
     'T': np.array([tsim, np.full((tsim.size,), 20)]).T,
     'f_prc': np.array([tsim, np.full((tsim.size,), 2)]).T,
     'LAI': np.array([tsim, np.full((tsim.size,), 2.5)]).T
     }

# Controlled inputs
u = {'f_Irg': 0}  # [mm d-1]

# Initialize module
water = Water(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0], tsim[-1])
y_water = water.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrieve the arrays from the dictionary of model outputs.
t_water = y_water['t']
L1 = y_water['L1']
L2 = y_water['L2']
L3 = y_water['L3']
# Plots
# TODO: Plot the state variables, (as L and theta) and flows.
# Include lines for the fc and pwp for each layer.
plt.plot(t_water, L1, label='L1')
plt.plot(t_water, L2, label='L2')
plt.plot(t_water, L3, label='L3')
plt.legend()
plt.show()
