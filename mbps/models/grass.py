# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Class for grass growth model from Grasim
"""
import numpy as np

from mbps.classes.module import Module
from mbps.functions.integration import fcn_euler_forward


class Grass(Module):
    ''' 
    Grass growth from Grasim model by Mohtar et al. (1997).
    Model time units in [d].
    
    Parameters
    ----------
    tsim : array
        A sequence of time points for the simulation [d]
    dt : scalar
        Time step for the numerical integration [d]
    x0 : dictionary of floats
        Initial conditions of the state variables
        
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'Ws'    [kgC m-2] Storage weight
        'Wg'    [kgC m-2] Structure weight
        ======  =============================================================
        
    p : dictionary of floats
        Model parameters
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'a'      [m2 kgC-1] structural specific leaf area
        'alpha'  [kgCO2 J-1] leaf photosynthetic efficiency
        'beta'   [d-1] senescence rate
        'k'      [-] extinction coefficient of canopy
        'm'      [-] leaf transmission coefficient
        'M'      [d-1] maintenance respiration coefficient
        'mu_m'   [d-1] max. structural specific growth rate
        'P0'     [kgCO2 m-2 d-1] max photosynthesis parameter
        'phi'    [-] photoshynthetic fraction for growth
        'Tmax'   [°C] maximum temperature for growth
        'Tmin'   [°C] minimum temperature for growth
        'Topt'   [°C] optimum temperature for growth
        'Y'      [-] structure fraction from storage
        'z'      [-] bell function power
        =======  ============================================================
    
    d : dictionary of 2D arrays 
        Model disturbances (required for method 'run'),
        of shape (len(t_d),2) for time and disturbance.
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'I_gl'   [J m-2 d-1] Global irradiance
        'T'      [°C] Environment average daily temperature
        'WAI'    [-] Water availability index
        =======  ============================================================        
    
    Returns
    -------
    y : dictionary of arrays
        Model outputs
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'Ws'     [kgC m-2] Storage weight
        'Wg'     [kgC m-2] Structure weight
        'LAI'    [-] Leaf area index
        'f_Gr'   [kgDM m-2 d-1] Graze dry matter
        'f_Hr'   [kgDM m-2 d-1] Harvest dry matter
        =======  ============================================================
        
    References
    ----------
    * Mohtar, R. H., Buckmaster, D. R., & Fales, S. L. (1997),
      A grazing simulation model: GRASIM A: Model development,
      Transactions of the ASAE, 40(5), 1483-1493.
    
    * Johnson, I. R., and J. H. M. Thornley (1983).
      Vegetative crop growth model incorporating leaf area expansion and
      senescence, and applied to grass, 
      Plant, Cell & Environment 6.9, 721-729.
    '''

    def __init__(self, tsim, dt, x0, p):
        Module.__init__(self, tsim, dt, x0, p)
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('f_P', 'f_SR', 'f_G', 'f_MR',
                       'f_R', 'f_S', 'f_Hr', 'f_Gr')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)

    def diff(self, _t, _x0):
        # -- Initial conditions
        Ws, Wg = _x0[0], _x0[1]

        # -- Physical constants
        theta = 12 / 44  # [-] CO2 to C (physical constant)

        # -- Model parameteres
        # TODO: Retrieve the rest of the model parameters
        # (this makes it easier to write the equations below)
        # TODO: Comment the units and short description for each parameter
        a = self.p['a']  # 40,[m2 kgC-1] structural specific leaf area
        alpha = self.p['alpha']  # [kgCO2 J-1] leaf photosynthetic efficiency
        beta = self.p['beta']  # [d-1] senescence rate
        k = self.p['k']  # [-] extinction coefficient of canopy
        m = self.p['m']  # [-] leaf transmission coefficient
        M = self.p['M']  # [d-1] maintenance respiration coefficient
        mu_m = self.p['mu_m']  # [d-1] max. structural specific growth rate
        P0 = self.p['P0']  # [kgCO2 m-2 d-1] max photosynthesis parameter
        phi = self.p['phi']  # [-] photoshynthetic fraction for growth
        Tmax = self.p['Tmax']  # [°C] maximum temperature for growth
        Tmin = self.p['Tmin']  # [°C] minimum temperature for growth
        Topt = self.p['Topt']  # [°C] optimum temperature for growth
        Y = self.p['Y']  # [-] structure fraction from storage
        z = self.p['z']  # [-] bell function power

        # -- Disturbances at instant _t
        I0, T, WAI = self.d['I0'], self.d['T'], self.d['WAI']
        _I0 = np.interp(_t, I0[:, 0], I0[:, 1])  # [J m-2 d-2] PAR
        _T = np.interp(_t, T[:, 0], T[:, 1])  # [°C] Environment temperature
        _WAI = np.interp(_t, WAI[:, 0], WAI[:, 1])  # [-] Water availability index

        # -- Controlled inputs
        f_Gr = self.u['f_Gr']  # [kgC m-2 d-1] Graze
        f_Hr = self.u['f_Hr']  # [kgC m-2 d-1] Harvest

        # -- Supporting equations
        # TODO: Fill in the supporting equations of the model
        # TODO: Comment the units and short description for each variable
        # - Mass
        W = Ws + Wg
        # - Temperature index [-]
        DTmax = max(Tmax - _T, 0)
        DTmin = max(_T - Tmin, 0)
        DTa = Tmax - Topt
        DTb = Topt - Tmin
        TI = ((DTmax / DTa) * (DTmin / DTb) ** (DTb / DTa)) ** z
        # - Photosynthesis
        LAI = a * Wg
        Pm = P0 * TI
        c1 = alpha * k * _I0 / (1 - m)
        P = Pm / k * np.log((c1 + Pm) / (c1 * np.exp(-k * LAI) + Pm))
        # - Flows
        # Photosynthesis [kgC m-2 d-1]
        f_P = _WAI * theta * phi * P
        # # Growth [kgC m-2 d-1]
        f_G = mu_m * Ws * Wg / (W)
        # Shoot respiration [kgC m-2 d-1]
        f_SR = (1 - Y) / Y * f_G
        # Maintenance respiration [kgC m-2 d-1]
        f_MR = M * Wg
        # Senescence [kgC m-2 d-1]
        f_S = beta * Wg
        gamma = 0
        # Recycling
        f_R = gamma * Wg

        # -- Differential equations [kgC m-2 d-1]
        # TODO: Write the differential equations based on the flow variables
        dWs_dt = f_P - f_SR - f_G - f_MR + f_R
        dWg_dt = f_G - f_R - f_S

        # -- Store flows [kgC m-2 d-1]
        idx = np.isin(self.t, _t)
        self.f['f_P'][idx] = f_P
        self.f['f_SR'][idx] = f_SR
        self.f['f_G'][idx] = f_G
        self.f['f_MR'][idx] = f_MR
        self.f['f_R'][idx] = f_R
        self.f['f_S'][idx] = f_S
        self.f['f_Hr'][idx] = f_Hr
        self.f['f_Gr'][idx] = f_Gr

        return np.array([dWs_dt, dWg_dt])

    def output(self, tspan):
        # Retrieve the required object properties
        dt = self.dt  # integration time step size
        diff = self.diff  # function with system of differential equations
        Ws0 = self.x0['Ws']  # initial condition
        Wg0 = self.x0['Wg']  # initial condiiton
        a = self.p['a']
        # Numerical integration
        # TODO: Call the Euler-forward integration function
        y0 = np.array([Ws0, Wg0])
        y_int = fcn_euler_forward(diff, tspan, y0, h=dt)
        # Model results
        # TODO: Retrieve the model outputs
        t = y_int['t']
        Ws = y_int['y'][0]
        Wg = y_int['y'][1]
        LAI = y_int['y'][1] / a
        return {
            't': t,  # [d] Integration time
            'Ws': Ws,  # [kgC m-2] Structure weight
            'Wg': Wg,  # [kgC m-2] Storage weight
            'LAI': LAI,  # [-] Leaf area index
            'f_Gr': self.u['f_Gr'],  # [kgC m-2 d-1] Graze dry matter
            'f_Hr': self.u['f_Hr'],  # [kgC m-2 d-1] Harvest dry matter
        }
