# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Generate and add noise to incoming data
 Project     : Simulation environment for BckTrk app
 File        : AWGN.py
 -----------------------------------------------------------------------------

   Description :


   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   26-Mai-2018  1.0      Rami      File created
 =============================================================================
 
"""

import numpy as np 
import Navigation.Coordinates as cord

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')

## the main processing function 
def noise_generator(params, positions_wm,noise_level):
    data_obj = cAWGN(params,noise_level)
    noise = data_obj.generate_noisy_signal_dist(positions_wm) ##if noise needed later
    return (data_obj.m_noisy_positions_wm,data_obj.m_noisy_positions_latlon,noise)

class cAWGN():
    ## Constructor
    def __init__(self, struct,noise_level):
        logger.debug("Initializing cAWGN")
        self.m_acquisition_length = struct['gps_freq_Hz']*struct['acquisition_time_sec']
        self.m_noisy_positions_latlon = np.zeros((2,self.m_acquisition_length))
        self.m_noisy_positions_wm = np.zeros((2,self.m_acquisition_length))
        self.m_noise_level = noise_level
        self.m_noise_std = struct['noise_std']
 
    ## noise generation based on the distance
    def generate_noisy_signal_dist(self, positions_wm):
        logger.debug("Generating noisy signal on distance")
        noise = np.transpose(np.random.multivariate_normal([0,0],[[1**2,0],[0,1**2]],self.m_acquisition_length))
        noise_dist = np.random.normal(loc=self.m_noise_level,scale=self.m_noise_std,size=self.m_acquisition_length)
        noise = noise*noise_dist
        self.m_noisy_positions_wm = positions_wm + noise
        self.m_noisy_positions_latlon = cord.generate_latlon_array(self.m_noisy_positions_wm)
        
        return noise
        