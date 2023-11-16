#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the Lotka-Volterra model
"""
import numpy as np
import matplotlib.pyplot as plt
from mbps.models.lotka_volterra import LotkaVolterra

plt.style.use('ggplot') # print(plt.style.available)

# Simulation time, initial conditions and parameters
tsim = np.arange(0, 365, 1)
dt = 7
x0 = {'prey':50, 'pred':50}
p = {'p1':1/30, 'p2':0.02/30, 'p3':0.01/30, 'p4':1/30}
# p = {'p1':1, 'p2':0.02, 'p3':0.01, 'p4':1}

# Initialize object
population = LotkaVolterra(tsim, dt, x0, p)

# Run model
tspan = (tsim[0],tsim[-1])
y = population.run(tspan)

# Retrieve results
t, t2 = y['t'], y['t2']
prey, prey2 = y['prey'], y['prey2']
pred, pred2 = y['pred'], y['pred2']

# Plot results
plt.figure(1)
plt.plot(t, prey, label='Preys')
plt.plot(t, pred, label='Preds')
plt.plot(t2, prey2, label='Preys_ivp')
plt.plot(t2, pred2, label='Preds_ivp')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$population\ [\#]$')
