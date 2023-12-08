# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Evaluation of the grass & water model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import LinearLocator
from mbps.models.grass_hav import Grass
from mbps.models.water_hav_irri import Water

highest_hav = 0
highest_hav_times = 0
highest_hav_irri_times = 0
highest_hav_ij = [0, 0]
hav_hist_amt = []
hav_hist_th = []
hav_hist_wg = []
irri_hist_water = []
irri_hist_times = []
hav_hist = []
plt.style.use('ggplot')
# hamt = np.linspace(0.05, 0.25, 21)
# hth = np.linspace(0.3, 0.65, 36)
hav_op_step = 50
hamt = np.linspace(0.05, 0.6, hav_op_step)
hth = np.linspace(0.1, 0.7, hav_op_step)
# hamt = np.linspace(0.05, 0.25, hav_op_step)
# hth = np.linspace(0.3, 0.65, hav_op_step)
for i in hamt:  # i, amt
    for j in hth:  # j,th
        if (j - i > 0.05):
            tsim = np.linspace(0, 365, int(365 / 5) + 1)  # [d]
            # Weather data (disturbances shared across models)
            # t_ini = '19950101'
            # t_end = '19960101'
            t_ini = '19950101'
            t_end = '19960101'
            t_weather = np.linspace(0, 365, 365 + 1)
            data_weather = pd.read_csv(
                '../data/etmgeg_260.csv',  # .. to move up one directory from current directory
                skipinitialspace=True,  # ignore spaces after comma separator
                header=47 - 3,  # row with column names, 0-indexed, excluding spaces
                usecols=['YYYYMMDD', 'TG', 'Q', 'RH'],  # columns to use
                index_col=0,  # column with row names from used columns, 0-indexed
            )

            # Grass data. (Organic matter assumed equal to DM) [gDM m-2]
            # Groot and Lantinga (2004)
            t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
            m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
            m_data = m_data / 1E3  # convert gDm to KgDM
            # This data was gathered in Wageningen 1995.

            # ---- Grass sub-model
            # Step size
            dt_grs = 1  # [d]x

            # Initial conditions
            # TODO: Specify suitable initial conditions for the grass sub-model
            x0_grs = {'Ws': 0, 'Wg': 0.02}  # [kgC m-2] according to the Mohtar et al. 1997
            # note that the Unit here is KgC m-2, meaning the weight of carbon content
            # Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
            p_grs = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area
                     'alpha': 2E-9,  # [kgCO2 J-1] leaf photosynthetic efficiency
                     'beta': 0.05,  # [d-1] senescence rate
                     'k': 0.5,  # [-] extinction coefficient of canopy
                     'm': 0.1,  # [-] leaf transmission coefficient
                     'M': 0.02,  # [d-1] maintenance respiration coefficient
                     'mu_m': 0.5,  # [d-1] max. structural specific growth rate
                     'P0': 0.432,  # [kgCO2 m-2 d-1] max photosynthesis parameter
                     'phi': 0.9,  # [-] photoshynth. fraction for growth
                     'Tmin': 0.0,  # [°C] maximum temperature for growth
                     'Topt': 20.0,  # [°C] minimum temperature for growth
                     'Tmax': 42.0,  # [°C] optimum temperature for growth
                     'Y': 0.75,  # [-] structure fraction from storage
                     'z': 1.33  # [-] bell function power
                     }
            # Model parameters adjusted manually to obtain growth
            # TODO: Adjust a few parameters to obtain growth.
            # Start by using the modifications from Case 1.
            # If needed, adjust further those or additional parameters
            p_grs['alpha'] = 2E-9 * 10
            p_grs['a'] = 21.35

            # Disturbances
            # PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
            T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
            I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

            T = T / 10  # [0.1 °C] to [°C] Environment temperature
            I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR

            d_grs = {'T': np.array([t_weather, T]).T,
                     'I0': np.array([t_weather, I0]).T,
                     }

            # Initialize module
            hav = [i, j, 0]  # amt, th, wg havrvested
            grass = Grass(tsim, dt_grs, x0_grs, p_grs, hav)

            # ---- Water sub-model
            dt_wtr = 1  # [d]

            # Initial conditions
            # TODO: Specify suitable initial conditions for the soil water sub-model
            x0_wtr = {'L1': 40, 'L2': 60, 'L3': 100, 'DSD': 2}  # 3*[mm], [d]
            # using the values from the Castellaro et al. 2009

            # Castellaro et al. 2009, and assumed values for soil types and layers
            p_wtr = {'S': 10,  # [mm d-1] parameter of precipitation retention
                     'alpha': 1.29E-6,  # [mm J-1] Priestley-Taylor parameter
                     'gamma': 0.68,  # [mbar °C-1] Psychrometric constant
                     'alb': 0.23,  # [-] Albedo (assumed constant crop & soil)
                     'kcrop': 0.90,  # [mm d-1] Evapotransp coefficient, range (0.85-1.0)
                     'WAIc': 0.75,  # [-] WDI critical, range (0.5-0.8)
                     'theta_fc1': 0.36,  # [-] Field capacity of soil layer 1
                     'theta_fc2': 0.32,  # [-] Field capacity of soil layer 2
                     'theta_fc3': 0.24,  # [-] Field capacity of soil layer 3
                     'theta_pwp1': 0.21,  # [-] Permanent wilting point of soil layer 1
                     'theta_pwp2': 0.17,  # [-] Permanent wilting point of soil layer 2
                     'theta_pwp3': 0.10,  # [-] Permanent wilting point of soil layer 3
                     'D1': 150,  # [mm] Depth of Soil layer 1
                     'D2': 250,  # [mm] Depth of soil layer 2
                     'D3': 600,  # [mm] Depth of soil layer 3
                     'krf1': 0.25,  # [-] Rootfraction layer 1 (guess)
                     'krf2': 0.50,  # [-] Rootfraction layer 2 (guess)
                     'krf3': 0.25,  # [-] Rootfraction layer 2 (guess)
                     'mlc': 0.2,  # [-] Fraction of soil covered by mulching
                     }

            # Disturbances
            # global irradiance [J m-2 d-1], environment temperature [°C],
            # precipitation [mm d-1], leaf area index [-].
            T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
            I_glb = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.
            f_prc = data_weather.loc[t_ini:t_end, 'RH'].values  # [0.1 mm d-1] Precipitation
            f_prc[f_prc < 0.0] = 0  # correct data that contains -0.1 for very low values

            T = T / 10  # [0.1 °C] to [°C] Environment temperature
            I_glb = I_glb * 1E4 / dt_wtr  # [J cm-2 d-1] to [J m-2 d-1] Global irradiance
            f_prc = f_prc / 10 / dt_wtr  # [0.1 mm d-1] to [mm d-1] Precipitation

            d_wtr = {'I_glb': np.array([t_weather, I_glb]).T,
                     'T': np.array([t_weather, T]).T,
                     'f_prc': np.array([t_weather, f_prc]).T,
                     }

            # Initialize module
            water = Water(tsim, dt_wtr, x0_wtr, p_wtr)

            # ---- Run simulation
            # Initial disturbance
            d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T

            # Iterator
            # (stop at second-to-last element, and store index in Fortran order)
            it = np.nditer(tsim[:-1], flags=['f_index'])
            for ti in it:
                # Index for current time instant
                idx = it.index
                # Integration span
                tspan = (tsim[idx], tsim[idx + 1])
                # Controlled inputs
                u_grs = {'f_Gr': 0,  # 0.3 * 15 * 1e-4 * 0.4
                         'f_Hr': 0}
                # f_Gr [kgDM m-2 d-1]  Graze dry matter,
                # 3 cows per one ha,value from farmer's knowledge, 15 [kgDM d-1] from Mohtar et al. 1997
                # Harvest dry matter
                u_wtr = {'f_Irg': 0}  # [mm d-1] Irrigation
                # Run grass model
                y_grs = grass.run(tspan, d_grs, u_grs)
                # Retrieve grass model outputs for water model
                d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
                # Run water model
                y_wtr = water.run(tspan, d_wtr, u_wtr)
                # Retrieve water model outputs for grass model
                d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

            # Retrieve simulation results
            t_grs, t_wtr = grass.t, water.t
            WsDM, WgDM, LAI = grass.y['Ws'] / 0.4, grass.y['Wg'] / 0.4, grass.y['LAI']
            L1, L2, L3 = water.y['L1'], water.y['L2'], water.y['L3'],
            WAI = water.y['WAI']
            if highest_hav < grass.hav_wg:
                highest_hav = grass.hav_wg
                highest_hav_ij = [i, j]
                highest_hav_times = grass.hav_times
                highest_hav_irri_times = water.irri_times
            hav_hist_amt.append(i)
            hav_hist_th.append(j)
            hav_hist_wg.append(grass.hav_wg)
            irri_hist_water.append(water.irri_water)
            irri_hist_times.append(water.irri_times)
        # hav_hist.append(i, j, grass.hav_wg)
        # print("harvest amount every time", i)
        # print("harvest threshold", j)
        # print("harvested", grass.hav_wg)
print("highest hav", highest_hav)
print("highest hav ij", highest_hav_ij)
print("highest hav_times", highest_hav_times)
print("higher_hav_irri_times", highest_hav_irri_times)
hav_hist_amt = np.asarray(hav_hist_amt)
hav_hist_th = np.asarray(hav_hist_th)
hav_hist_wg = np.asarray(hav_hist_wg)

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection='3d')

# defining axes
c = hav_hist_wg
p = ax.scatter(hav_hist_amt, hav_hist_th, hav_hist_wg, c=c)

# syntax for plotting
ax.set_title('harvested yield -- harvest amount -- harvest threshold')
ax.set_xlabel('harvest amount' + r'$\ [kgDM m^{-2}]$')
ax.set_ylabel('harvest threshold' + r'$\ [kgDM m^{-2}]$')
ax.set_zlabel('harvested yield' + r'$\ [kgDM m^{-2}]$')
cbar = fig.colorbar(p)
cbar.ax.set_ylabel('harvested yield' + r'$\ [kgDM m^{-2}]$', labelpad=20, loc="center", rotation=270)
plt.show()
