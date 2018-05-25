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
import pygeodesy as geo 

## Bring the logger
import logging
logger = logging.getLogger(__name__)

## the main processing function 
def random_2d_path_generator(params):
    data_obj = cRandomPath(params)
    data_obj.initialize_position()
    data_obj.generate_path()
    path = data_obj.m_positions
    return path

class cRandomPath:
    ## Constructor
    def __init__(self,struct):
        self.m_localStruct = struct
        # Controls whether we want to use our random seed or let numpy initialize the randomstate "randomly"
        self.m_use_random_seed = struct['use_random_seed']
        self.m_random_seed = struct['random_seed']
        self.m_acquisition_length = struct['gps_freq_Hz']*struct['acquisition_time_sec']
        self.m_positions = np.zeros((2,self.m_acquisition_length))
        self.m_stepsize = struct['stepsize']
        if self.m_use_random_seed :
            np.random.seed(self.m_random_seed)
        np.random.seed(self.m_random_seed)
            
        
    ## Set randomly initial position (longitute and lattitude)
    def initialize_position(self) :
        lat = np.random.uniform(-85.051,85.051) # Generates random lattitude in degrees limits set by webmercator
        lon = np.random.uniform(-180.0,180.0) #Generates random longitude in degrees
        self.m_initial_position_latlon = geo.ellipsoidalVincenty.LatLon(lat,lon)
        
    def generate_path(self) :
        """generates a path of (x,y) coordinates"""
        initial_position_wm = geo.toWm(self.m_initial_position_latlon)
        self.m_positions[:,0] = np.array([initial_position_wm.x,initial_position_wm.y])
        for i in range(self.m_acquisition_length-1):
            theta = np.random.uniform(0,2*np.pi)
            self.m_positions[:,i+1] = self.m_positions[:,i] + self.m_stepsize*np.array([np.cos(theta),np.sin(theta)])
        
        

    
    
    
        