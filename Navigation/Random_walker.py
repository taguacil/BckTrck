# %-*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Random walker
 Project     : Simulation environment for BckTrk app
 File        : Random_walker.py
 -----------------------------------------------------------------------------

   Description :

   This file generates a random 2d walk over a grid, can only move in cardinal directions, 
   the stepsize in normalized to 1 to avoid numerical precision problems, 
   returns a 2d array with the x-positions on one axis and the y-positions on the other 
   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   26-Apr-2018  1.0      Rami      File created
 =============================================================================
 
"""
## Python library import
import numpy as np 

## Bring the logger
import logging
logger = logging.getLogger(__name__)

## the main processing function 
def random_2d_path_generator (params):
    data_obj = cRandomPath (params)
    
class cRandomPath:
    ## Constructor
    def __init__(self,struct):
        self.m_localStruct = struct
        # Controls whether we want to use our random seed or let numpy initialize the randomstate "randomly"
        self.m_use_random_seed = struct['use_random_seed']
        self.m_random_seed = struct['random_seed']
        self.m_acquisition_length = struct['gps_freq_Hz'] * struct['acquisition_time_sec']
        self.m_positions = np.zeros ((2,self.m_aquisition_length))
        if self.m_use_random_seed :
            np.random.seed(self.m_random_seed)
            
    ## An example of required methods    
    
    ## Set randomly initial position (longitute and lattitude)
    
    ## Generate randomly the bearings(radian) and distances (meter)
    
    ## Compute new position with equations
    
    ## Calculate distance and bearing between 2 positions (optional, might be helpful later)
    
        