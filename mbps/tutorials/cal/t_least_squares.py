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
t_sim = np.linspace(0.0, 365.0, 365 + 1)  # [d]

# Initial condition
# TODO: Based on the data below, propose a sensible value for the initial mass
m0 = 0.1           # [kgDM m-2] initial mass

# Organic matter (assumed equal to DM) measured in Wageningen 1995 [gDM m-2]
# Groot and Lantinga (2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([0.156, 0.198, 0.333, 0.414, 0.510, 0.640, 0.663, 0.774])
# TODO: this file uses the analytical solution for logistic growth.
# Adjust t_data so that it matches t_sim with t0=0.
t_data = ???


# Define a function to simulate the model output of interest (m)
# as a function of the estimated parameter array 'p'
def fcn_y(p):
    # Model parameters (improved iteratively)
    r, K = p[0], p[1]   # [d-1], [kgDM m-2] model parameters
    # Model output (analytical solution of logistic growth model)
    # TODO: define 'm' based on the analytical solution for logistic growth,
    # using t_sim.
    m = ???  # [kgDM m-2]
    return m

# Define a function to calculate the residuals: e(k|p) = z(k)-y(k|p)
# Notice that m_k must be interpolated for measurement instants t_data
def fcn_residuals(p):
    # TODO: calculate m from the model output function 'fcn_y' defined above
    m = ???
    # TODO: create an interpolation function using Scipy interp1d,
    # based on the simulation time and mass arrays
    f_interp = ???
    # TODO: call the interpolation function for the measurement instants
    m_k = f_interp(???)
    # TODO: Calculate the residuals (err)
    err = ???
    return err
    
# Model calibration: least_squares
# TODO: Define an array for the initial guess of r and K (mind the order)
p0 = np.array([???, ???])                   # Initial parameter guess
# TODO: Call the Scipy method least_squares
y_lsq = least_squares(???, ???)              # Minimize sum [ e(k|p) ]^2

# Retrieve the calibration results (parameter estimates)
# TODO: Once the code runs and you obtain y_lsq (a dictionary),
# look into y_lsq and identify its elements. Uncomment the line below
# and retrieve the parameter estimates.
#p_hat = y_lsq[???]

# Simulate the model with initial guess (p0) and estimated parameters (p_hat)
# TODO: define variables m_hat0 (mass from initial parameter guess),
# and m_hat (mass from estimated parameters)


# Plot results
# TODO: Make a plot for the growth of m_hat0 (dashed line),
# m_hat (continuous line), and mdata (no line, markers)

# -- EXERCISE 1.2 --
# Calibration accuracy

# Jacobian matrix
# TODO: Retrieve the sensitivity matrix (Jacobian) J from y_ls.

# Residuals
# TODO: Retrieve the residuals from y_ls

# Variance of residuals
# TODO: Calculate the variance of residuals

# Covariance matric of parameter estimates
# TODO: Calculate the covariance matrix

# Standard deviations of parameter estimates
# TODO: Calculate the variance and standard error

# Correlation coefficients of parameter estimates
# TODO: Calculate the correlation coefficients
# (you can use a nested for-loop, i.e., a for-loops inside another)


# References

# Groot, J. C., & Lantinga, E. A. (2004). An object-oriented model of the
#  morphological development and digestibility of perennial ryegrass.
# Ecological Modelling, 177(3-4), 297-312.

# Bouman, B.A.M., Schapendonk, A.H.C.M., Stol, W., van Kralingen, D.W.G.
#  (1996) Description of the growth model LINGRA as implemented in CGMS.
#  Quantitative Approaches in Systems Analysis no. 7
#  Fig. 3.4, p. 35