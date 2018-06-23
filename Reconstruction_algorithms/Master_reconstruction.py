# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Main reconstruction script
 Project     : Simulation environment for BckTrk app
 File        : Master_reconstruction.py
 -----------------------------------------------------------------------------

   Description :

 
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   21-Jun-2018  1.0      Ramy      File created
 =============================================================================
 
"""
## Python library import
import numpy as np

## Logging
import logging
logger = logging.getLogger('BckTrk')

## User-defined library import
from .Lasso_reconstruction import lasso_algo

## Helper functions
def identify_algorithms(params):
    temp = []
    if params['RCT_ALG_LASSO']["bReconstruct_lasso"] : 
        temp.append("Lasso")
    return temp

## The main processing function
def reconstructor(params, path) :
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')
    
    reconstructed_paths = {}
    mean                = np.repeat(np.expand_dims(np.mean(path, axis=1),axis=1),params['acquisition_length'],axis=1)
    var                 = np.repeat(np.expand_dims(np.var(path, axis=1),axis=1),params['acquisition_length'],axis=1)
    normalized_path     = (path-mean)/np.sqrt(var)
    
    if params['RCT_ALG_LASSO']["bReconstruct_lasso"] :
        temp                         = np.array([lasso_algo(params, normalized_path[0]),lasso_algo(params, normalized_path[1])])
        rescaled_temp                = np.sqrt(var)*temp + mean
        reconstructed_paths["Lasso"] = rescaled_temp
        
    return reconstructed_paths
