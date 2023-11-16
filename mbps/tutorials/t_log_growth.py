# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for logistic growth model
"""
import numpy as np
import matplotlib.pyplot as plt
from mbps.models.log_growth import LogisticGrowth

plt.style.use('ggplot') # print(plt.style.available)

# Simulation time array
tsim = np.linspace(0, 10, 100+1) # [d]

# Initialize model object
dt = 1.0                # [d] time-step size
x0 = {'m':1.0}          # [gDM m-2] initial conditions
p = {'r':1.2, 'K':100}   # [d-1], [gDM m-2] model parameters
lg = LogisticGrowth(tsim, dt, x0, p)

# Run model
tspan = (tsim[0],tsim[-1])
y = lg.run(tspan)

# Analytical solution
m0, r, K = x0['m'], p['r'], p['K']
m_anl = K/(1 + (K-m0)/m0*np.exp(-r*tsim))

# Plot results
plt.figure(1)
plt.plot(tsim, m_anl, label='analytical', color='purple')
plt.plot(y['t'], y['m'], label='Euler-forward', color='green')
plt.plot(y['t_rk'], y['m_rk'], label='Runge-Kutta 4th order',
         linestyle='--', color='orange')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$mass\ [gDM\ m^{-2}]$')

# Error assessment
e_ef = y['m'] - np.interp(y['t'], tsim, m_anl)
e_rk = y['m_rk'] - np.interp(y['t_rk'], tsim, m_anl)
plt.figure(2)
plt.plot(y['t'], e_ef, label='error EF', color='green')
plt.plot(y['t_rk'], e_rk, label='error RK4', linestyle='--', color='orange')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$error\ [gDM\ m^{-2}]$')
plt.show()

