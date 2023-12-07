from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbps.models.water_sol import Water
from mbps.models.grass_sol import Grass
from mbps.functions.calibration import fcn_residuals, fcn_accuracy
from mbps.functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')
# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

#### -- Data --
# Simulation time
tsim = np.linspace(0, 365, 365 + 1)  # [d]

# Weather data (disturbances)
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
# (Groot and Lantinga, 2004)
t_data = np.array([107, 114, 122, 129, 136, 142, 149, 156])
m_data = np.array([156., 198., 333., 414., 510., 640., 663., 774.])
m_data = m_data / 1E3

# ---- Grass sub-model
# Step size
dt_grs = 1  # [d]x

# Initial conditions
# TODO: Specify suitable initial conditions for the grass sub-model
x0_grs = {'Ws': 0, 'Wg': 0.02}  # [kgC m-2] according to the Mohtar et al. 1997
# note that the Unit here is KgC m-2, meaning the weight of carbon content
# Model parameters (as provided by Mohtar et al. 1997 p.1492-1493)
p_grs = {'a': 40.0,  # [m2 kgC-1] structural specific leaf area. smaller a, later growth
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
# p_grs['alpha'] = 1.29E-8
# p_grs['a'] = 36
# p_grs['alpha'] = 1.5E-9 * 10
# p_grs['phi'] = 0.8
# gives very good result
# p_grs['a'] = 40
p_grs['alpha'] = 2.0E-9 * 10

# Disturbances
# PAR [J m-2 d-1], environment temperature [°C], leaf area index [-]
T = data_weather.loc[t_ini:t_end, 'TG'].values  # [0.1 °C] Env. temperature
I_gl = data_weather.loc[t_ini:t_end, 'Q'].values  # [J cm-2 d-1] Global irr.

T = T / 10  # [0.1 °C] to [°C] Environment temperature
I0 = 0.45 * I_gl * 1E4 / dt_grs  # [J cm-2 d-1] to [J m-2 d-1] Global irr. to PAR
# I0 = I_gl * 1E4 / dt_grs
d_grs = {'T': np.array([t_weather, T]).T,
         'I0': np.array([t_weather, I0]).T,
         }

# Initialize module
grass = Grass(tsim, dt_grs, x0_grs, p_grs)

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


# adapt the function fnc_y from grass calibration to grasswater calibration
def fnc_y(p0):
    # Reset initial conditions
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()
    grass.p['alpha'] = p0[0]
    grass.p['a'] = p0[1]

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
        u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
        u_wtr = {'f_Irg': 0}  # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

    return grass.y['Wg'] / 0.4


# Run calibration function
# TODO: Specify the initial guess for the parameter values
# These can be the reference values provided by Mohtar et al. (1997),
# You can simply call them from the dictionary p.
p0 = np.array([p_grs['alpha'], p_grs['a']])  # Initial guess , p_wtr['kcrop']
# p0 = np.array([p_grs['phi'], p_grs['alpha'], p_wtr['kcrop'], p_grs['Y']])  # Initial guess
# Parameter bounds
# TODO: Specify bounds for your parameters, e.g., efficiencies lie between (0,1).
# Use a tuple of tuples for min and max values:
# ((p1_min, p2_min, p3_min), (p1_max, p2_max, p3_max))
bnds = ((1.9E-9 * 10, 20), (2.1E-9 * 10, 50))
# Call the lest_squares function.
# Our own residuals function takes the necessary positional argument p0,
# and the additional arguments fcn_y, t ,tdata, ydata.
y_ls = least_squares(fcn_residuals, p0, bounds=bnds,
                     args=(fnc_y, grass.t, t_data, m_data), ftol=1E-12,
                     kwargs={'plot_progress': True})
# max_nfev=100000,
# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)
# Run calibrated simulation
# TODO: Retrieve the parameter estimates from
# the output of the least_squares function
p_hat = y_ls['x']
# TODO: Run the model output simulation function with the estimated parameters
# (this is the calibrated model output)
WgDM_hat = fnc_y(p_hat)

#### -- Plot results --
# TODO: Make one figure comparing the calibrated model against
# the measured data
plt.figure(1)
plt.plot(tsim, WgDM_hat, label='m_hat')
plt.plot(t_data, m_data, label='m_data', marker="o", linestyle='None')
plt.xlabel('Time [d]')
plt.ylabel('grass biomass' + r'$[kg\cdot m^{-2}]$')
plt.legend()
plt.show()

# ---- Uncertainty Analysis
# Monte Carlo simulations
n_sim = 100  # number of simulations
# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size, n_sim), np.nan)
for j in range(n_sim):
    # Reset initial conditions
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()
    # Sample random parameters

    grass.p['alpha'] = rng.normal(p_hat[0], y_calib_acc['sd'][0])
    grass.p['a'] = rng.normal(p_hat[1], y_calib_acc['sd'][1])
    # Model output
    d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T
    it = np.nditer(tsim[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Integration span
        tspan = (tsim[idx], tsim[idx + 1])
        # Controlled inputs
        u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
        u_wtr = {'f_Irg': 0}  # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])
    # Retrieve results and store in array of outputs
    m_arr[:, j] = grass.y['Wg'] / 0.4

plt.figure(2)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.5, 0.68, 0.95])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('cummulative mass ' + r'$[kgDM\ m^{-2}]$')
plt.show()
