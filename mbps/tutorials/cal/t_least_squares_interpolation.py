# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the use of the least_squares method
Exercises 1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import numpy.linalg as LA

plt.style.use('ggplot')

# -- EXERCISE 1.1 --
# Simulation time array
# TODO: define an array for a period that matches the period of measurements
# Notice that the analytical solution is based on the running time. Therefore,
# it's better to start the simulation from t0=0.
t_sim = np.linspace(0.0, 49, 49 + 1)  # [d]

# Initial condition
# TODO: Based on the data below, propose a sensible value for the initial mass
m0 = 0.1  # [kgDM m-2] initial mass

# Organic matter (assumed equal to DM) measured in Wageningen 1995 [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([0.156, 0.198, 0.333, 0.414, 0.510, 0.640, 0.663, 0.774])
# TODO: this file uses the analytical solution for logistic growth.
# Adjust t_data so that it matches t_sim with t0=0.
f = interp1d(t_data - 107, m_data)
t_data = f(t_sim)
print(t_sim)
print(t_data)
