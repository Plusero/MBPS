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
dt = 1.0  # [d] time-step size
x0 = {'prey': 50, 'pred': 50}  # populations [preys, preds]
# Model parameters
# p1 [d-1], p2 [pred-1 d-1], p3 [prey-1 d-1], p4 [d-1]
# p = {'p3': 0.01 / 30, 'p4': 1.0 / 30}
p = {'p2': 0.02 / 30, 'p4': 1.0 / 30}
# Initialize object
lv = LotkaVolterra(tsim, dt, x0, p)

# Data
t_data = np.array([60, 120, 180, 240, 300, 360])
y_data = np.array([[96, 191, 61, 83, 212, 41],  # [preys]
                   [18, 50, 64, 35, 40, 91]]).T  # [preds]
print("shape y_data", y_data.shape)


# Define function to simulate model based on estimated array 'p0'.
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. Estimate p1 and p3
def fcn_y(p0):
    # Reset initial conditions
    lv.x0 = x0.copy()
    # Reassign parameters from array p0 to object
    lv.p['p1'] = p0[0]
    lv.p['p3'] = p0[1]
    # Simulate the model
    y = lv.run(tspan)
    # Retrieve result (model output of interest)
    # Note: For computational speed in the least squares routine,
    # it is best to compute the residuals based on numpy arrays.
    # We use rows for time, and columns for model outputs.
    # TODO: retrieve the model outputs into a numpy array for populations 'pop'
    # pop = y['prey', 'pred']
    # pop = y['x']
    pop = np.stack((y['prey'], y['pred']), axis=-1)
    print("pop.shape", pop.shape)
    return pop


# Run calibration function
# -- Exercise 3.1. Estimate p1 and p2
# -- Exercise 3.2. estimate p1 and p3
p0 = np.array([1 / 30, 0.01 / 30])  # Initial guess
y_ls = least_squares(fcn_residuals, p0,
                     bounds=([1E-6, 1E-6], [np.inf, np.inf]),
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
plt.plot(tsim, y_hat[:, 0], label='prey')
plt.plot(tsim, y_hat[:, 1], label='pred')
plt.plot(t_data, y_data[:, 0], 'o', label='data prey')
plt.plot(t_data, y_data[:, 1], 'x', label='data pred')
plt.legend()
plt.show()

J = y_ls['jac']
# Residuals
# TODO: Retrieve the residuals from y_ls
res = y_ls['fun']
# Variance of residuals
# TODO: Calculate the variance of residuals
N_ins = len(tsim)  # total number of measurements at instants
n_p = len(p0)  # the number of model parameters.
var_errr = 1 / (N_ins - n_p) * (res.T @ res)
# Covariance matric of parameter estimates
# TODO: Calculate the covariance matrix
cov_matrix = (res.T @ res) / (N_ins - n_p) * (np.transpose(J) @ J) ** -1
# Standard deviations of parameter estimates
# TODO: Calculate the variance and standard error
variance = cov_matrix.diagonal
# variance_i = np.fliplr(cov_matrix).diagonal()
# s_i = np.sqrt(variance_i)
s = np.sqrt(cov_matrix.diagonal())
# Correlation coefficients of parameter estimates
# TODO: Calculate the correlation coefficients
# (you can use a nested for-loop, i.e., a for-loops inside another) why tho??
# for i in
cc = cov_matrix[0][1] ** 2 / (s[0] * s[1])
print("cov_matrix", cov_matrix)
print("s", s)
print("cc", cc)
