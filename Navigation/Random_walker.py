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
import Navigation.Coordinates as cord

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')

## the main processing function 
def random_2d_path_generator(params):
    data_obj = cRandomPath(params)
    data_obj.initialize_position()
    data_obj.generate_wm_path()

    path_wm = data_obj.m_positions_wm
    path_latlon = data_obj.m_positions_latlon
    return (path_wm,path_latlon)

class cRandomPath:
    ## Constructor
    def __init__(self,struct):
        logger.debug("Initializing cRandomPath")
        # Controls whether we want to use our random seed or let numpy initialize the randomstate "randomly"
        self.m_acquisition_length   = struct['acquisition_length']
        self.m_stepsize             = struct['stepsize']
        self.m_use_static_init      = struct['bUse_static_init']
        self.m_static_lat           = struct['staticLat']
        self.m_static_lon           = struct['staticLon']
        
        self.m_positions_wm         = np.zeros((2,self.m_acquisition_length))
        self.m_positions_latlon     = np.zeros((2,self.m_acquisition_length))

        
    ## Set randomly initial position (longitute and lattitude)
    def initialize_position(self) :
        logger.debug("Generating initial position in LatLon")
        if self.m_use_static_init:
            lat = self.m_static_lat
            lon = self.m_static_lon
            logger.debug("Latitude and longitude used respectively <%f> <%f>",lat,lon)
        else:
            lat = np.random.uniform(-85.051,85.051) # Generates random lattitude in degrees limits set by webmercator
            lon = np.random.uniform(-180.0,180.0) #Generates random longitude in degrees
        
        self.m_initial_position_latlon = geo.ellipsoidalVincenty.LatLon(lat,lon)
        
    ## Generate data in cartesian plane
    def generate_wm_path(self) :
        logger.debug("Generating a path of WM coordinates")
        initial_position_wm = geo.toWm(self.m_initial_position_latlon)
        bearing             = np.random.uniform(0,2*np.pi,(self.m_acquisition_length))
     
        self.m_positions_wm[:,0] = np.array([initial_position_wm.x,initial_position_wm.y])
        for i in range(self.m_acquisition_length-1):
            self.m_positions_wm[:,i+1] = self.m_positions_wm[:,i] + self.m_stepsize*np.array([np.cos(bearing[i]),np.sin(bearing[i])])
        
        self.m_positions_latlon = cord.generate_latlon_array(self.m_positions_wm)

        