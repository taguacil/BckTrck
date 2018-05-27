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
import pygeodesy as geo 
import Navigation.Coordinates as cord

class cInstrument():
    
    def __init__(self, struct):
        self.m_localStruct = struct
        self.m_use_random_seed = struct['use_random_seed']
        self.m_random_seed = struct['random_seed']
        self.m_acquisition_length = struct['gps_freq_Hz']*struct['acquisition_time_sec']
        self.m_true_positions_latlon = np.empty((2,self.m_acquisition_length),dtype=object)
        self.m_true_positions_wm = np.empty((2,self.m_acquisition_length),dtype=object)
        self.m_noisy_positions_latlon = np.empty((2,self.m_acquisition_length),dtype=object)
        self.m_noisy_positions_wm = np.empty((2,self.m_acquisition_length),dtype=object)
        self.m_noise_level = struct['noise_level']
        if self.m_use_random_seed :
            np.random.seed(self.m_random_seed)
        
    def generate_true_signal(self, positions):
        self.m_true_positions_wm = cord.convert_xy_to_wm(positions)
        self.m_true_positions_latlon = cord.convert_wm_to_elatlon(self.m_true_positions_wm)
        
    def generate_noisy_signal(self, positions):
        noise = np.transpose(np.random.multivariate_normal([0,0],[[self.m_noise_level**2,0],[0,self.m_noise_level**2]],self.m_acquisition_length))
        self.m_noisy_positions_wm = cord.convert_xy_to_wm(positions + noise)
        self.m_noisy_positions_latlon = cord.convert_wm_to_elatlon(self.m_noisy_positions_wm)
        return noise
        