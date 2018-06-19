# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Domain transforms script
 Project     : Simulation environment for BckTrk app
 File        : transform.py
 -----------------------------------------------------------------------------

   Description :

 
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   19-Jun-2018  1.0      Taimir      File created
 =============================================================================
 
"""
## Python library import
import numpy as np 
import scipy.fftpack as ft

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')

## the main processing function 
def transforms (params,paths_latlon_noisy):
    data_obj = cTransforms (params,paths_latlon_noisy)
    if params['dctTransform']:
        return(data_obj.transform_dct())
    
class cTransforms:
    ## Constructor
    def __init__(self,struct,paths_latlon_noisy):
        self.m_latlonpath = paths_latlon_noisy
        self.m_acquisition_length = struct['gps_freq_Hz']*struct['acquisition_time_sec']
        
    ## Write dictionary into pickle format to txt file    
    def transform_dct(self) :
        transformed_path= np.zeros((2,self.m_acquisition_length))
        transformed_path[0,:] = ft.dct(self.m_latlonpath[0,:])
        transformed_path[1,:] = ft.dct(self.m_latlonpath[1,:])
        return transformed_path
