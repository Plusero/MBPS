#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Your team names

Class for disease SIR model
"""
import numpy as np

from mbps.classes.module import Module
from mbps.functions.integration import fcn_euler_forward

class SIR(Module):
    """ Module for disease spread
    
    Parameters
    ----------
    Add here the parameters required to initialize your object
    
    Returns
    -------
    Add here the model outputs returned by your object
    
    """
    # Initialize object. Inherit methods from object Module
    # TODO: fill in the required code
    def __init__(???):
        ???
    
    # Define system of differential equations of the model
    # TODO: fill in the required code.
    '''Explanation
    Notice that for the function diff, we use _t and _y0.
    This underscore (_) notation is used to define internal variables,
    which are only used inside the function.
    It is useful here to represent the fact that _t and _y0 are changing
    iteratively, every time step during the numerical integration
    (in this case, called by 'fcn_euler_forward')
    '''
    def diff(self,_t,_y0):
        # State variables
        ???
        # Parameters
        ???
        # Differential equations
        ???
        return ???
        
    # Define model outputs from numerical integration of differential equations
    # This function is called by the Module method 'run'.
    # The model does not use disturbances (d), nor control inputs (u).
    # TODO: fill in the required code
    def output(self,tspan):
        # Retrieve object properties
        ???
        # initial conditions
        ???
        # Numerical integration
        # (for numerical integration, y0 must be numpy array,
        # even for a single state variable)
        y0 = np.array([???, ???, ???,])
        y_int = fcn_euler_forward(???)
        # Retrieve results from numerical integration output
        t = ???
        s, i, r = ???
        return {'t':t, 's':s, 'i':i, 'r':r}
