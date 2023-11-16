# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- write your team names here --

Tutorial for the calibration of the Lotka-Volterra model
Exercise 3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from mbps.models.lotka_volterra import LotkaVolterra
from mbps.functions.calibration import fcn_residuals, fcn_accuracy

plt.style.use('ggplot')

# Simulation time array
tsim = np.arange(0, 365, 1)
tspan = (tsim[0], tsim[-1])

# Initialize reference object
dt = 1.0                    # [d] time-step size
x0 = {'prey':50, 'pred':50} # populations [preys, preds]
# Model parameters
# p1 [d-1], p2 [pred-1 d-1], p3 [prey-1 d-1], p4 [d-1]
p = {'p3':0.01/30, 'p4':1.0/30}
# Initialize object
lv = LotkaVolterra(tsim, dt, x0, p)

# Data
t_data = np.array([60, 120, 180, 240, 300, 360])
y_data = np.array([[96, 191, 61, 83, 212, 41],  # [preys]
                   [18, 50, 64, 35, 40, 91]]).T # [preds]

# Define function to simulate model based on estimated array 'p0'.
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. Estimate p1 and p3
def fcn_y(p0):
    # Reset initial conditions
    lv.x0 = x0.copy()
    # Reassign parameters from array p0 to object
    lv.p['p1'] = p0[0]
    lv.p['p2'] = p0[1]
    # Simulate the model
    y = lv.run(tspan)
    # Retrieve result (model output of interest)
    # Note: For computational speed in the least squares routine,
    # it is best to compute the residuals based on numpy arrays.
    # We use rows for time, and columns for model outputs.
    # TODO: retrieve the model outputs into a numpy array for populations 'pop'
    pop = ???
    return pop

# Run calibration function
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. estimate p1 and p3
p0 = np.array([0.01/30, 0.01/30]) # Initial guess
y_ls = least_squares(fcn_residuals, p0,
                     bounds = ([1E-6, 1E-6], [np.inf, np.inf]),
                     args=(fcn_y, lv.t, t_data, y_data),
                     )

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Run model output function with the estimated parameters
p_hat_arr = y_ls['x']
y_hat = fcn_y(p_hat_arr)

# Plot calibrated model
# -- Exercise 3.1 and 3.2
# TODO: plot the model output based on the estimated parameters,
# together with the data.
plt.figure('Calibrated model and data')
