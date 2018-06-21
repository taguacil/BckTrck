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
    if params["reconstruct_lasso"] : 
        temp.append("Lasso")
    return temp

## The main processing function
def reconstructor(params, path) :
    """returns dictionary of different path reconstructions using different algorithms"""
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')
    reconstructed_paths = {}
    if params["reconstruct_lasso"] :
        temp = np.array([lasso_algo(params, path[0]),lasso_algo(params, path[1])])
        reconstructed_paths["Lasso"] = temp
    return reconstructed_paths
