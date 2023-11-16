# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for disease model (SIR)
"""
import numpy as np
import matplotlib.pyplot as plt
from mbps.models.sir import SIR

plt.style.use('ggplot') # print(plt.style.available)

# Simulation time array
tsim = np.linspace(0, 365, 365+1) # [d]

# Initialize model object
dt = 1.0                            # [d] time-step size
x0 = {'s':0.99, 'i':0.01, 'r':0.0}  # [gDM m-2] initial conditions
p = {'beta':0.1,'gamma':0.02}       # [d-1]x2 model parameters
# p = {'beta':0.5,'gamma':0.02}       # [d-1]x2 model parameters (initial)
disease = SIR(tsim, dt, x0, p)

# Run model
tspan = (tsim[0],tsim[-1])
y = disease.run(tspan)

# Plot results
plt.figure(1)
plt.plot(y['t'], y['s'], label='S')
plt.plot(y['t'], y['i'], label='I')
plt.plot(y['t'], y['r'], label='R')
plt.xlabel('time ' + r'$[d]$')
plt.ylabel('population fraction ' + r'$[-]$')
plt.legend()
