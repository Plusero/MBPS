# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   -- Fill in your team names --

Tutorial: Soil water model analysis.
1. Slope of saturation vapour pressure.
2. Reference evapotranspiration.
"""
# TODO: import the required packages
import numpy as np
import matplotlib.pyplot as plt

# TODO: specify the matplotlib style
plt.style.use('ggplot')

# Measured saturation vapour pressure
# TODO: define arrays for the measured data T and Pvs.
# Specify the units in comments
T_data = np.array([10, 20, 30, 40])  # [°C]
T_data_Kelvin = np.array([10, 20, 30, 40]) + 273.15  # [K]
Pvs_data = np.array([12.28, 23.39, 42.46, 73.84])  # [mbar]

# Air temperature [K]
# TODO: define an array for sensible values of T
T = np.linspace(0, 40, 401)

# Model parameteres and variables
alpha = 1.291  # [mm MJ-1 (m-2)] Priestley-Taylor parameter
gamma = 0.68  # [mbar °C-1] Psychrometric constant
Irr_gl = 18.0  # [MJ m-2 d-2] Global irradiance
alb = 0.23  # [-] albedo (crop)
# Rn = 0.408 * Irr_gl * 1 - (alb)  # [MJ m-2 d-1] Net radiation
Rn = 1e6
# Model equations
# TODO: Define variables for the model
# Exercise 1. Pvs, Delta 
# Exercise 2. ET0
Pvs = np.exp(21.3 - 5304 / (T_data_Kelvin)) - 2.2
Pvs_10 = np.exp(21.3 - 5304 / T_data_Kelvin) - 0.75814
Pvs_20 = np.exp(21.3 - 5304 / T_data_Kelvin) - 1.31174
Pvs_30 = np.exp(21.3 - 5304 / T_data_Kelvin) - 2.40738
Pvs_40 = np.exp(21.3 - 5304 / T_data_Kelvin) - 4.60754
ET0 = alpha * Rn * (Pvs / (Pvs + gamma))
# Relative error
# TODO: Calculate the average relative error
# Tip: The numpy functions np.isin() or np.where() can help you retrieve the
# modelled values for Pvs at the corresponding value for T_data.
error = np.abs(Pvs - Pvs_data) / Pvs_data
print('Average relative error: {:.2f}%'.format(np.mean(error) * 100))
# Figures
# TODO: Make the plots
# Exercise 1. Pvs vs. T and Delta vs. T,
# Exercise 2. ET0 vs. T
plt.figure(1)
plt.plot(T_data, Pvs, label='Model_avg')
plt.plot(T_data, Pvs_10, label='Model_10')
plt.plot(T_data, Pvs_20, label='Model_20')
plt.plot(T_data, Pvs_30, label='Model_30')
plt.plot(T_data, Pvs_40, label='Model_40')
plt.plot(T_data, Pvs_data, 'o', label='Experirment')
plt.legend()
plt.figure(2)
plt.plot(T_data_Kelvin, ET0, 'o', label='Model')
plt.legend()
plt.show()
