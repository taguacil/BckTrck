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
# Python library import
import numpy as np

# User-defined library import
from .Lasso_reconstruction import lasso_algo
from .BFGS_reconstruction import bfgs_algo
from Helper_functions.framework_error import CFrameworkError
from NeuralNetworks.NN import CNeuralNetwork

# Logging
import logging

logger = logging.getLogger('BckTrk')

# Helper functions
def identify_algorithms(params):
    temp = []
    if params['RCT_ALG_LASSO']["bReconstruct_lasso"]:
        temp.append("Lasso")
    if params['RCT_ALG_BFGS']["bReconstruct_bfgs"]:
        temp.append("BFGS")
    return temp


# The main processing function
def reconstructor(params, path):
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')

    mean = np.repeat(np.expand_dims(np.mean(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    var = np.repeat(np.expand_dims(np.var(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    normalized_path = (path - mean) / np.sqrt(var)

    if params['RCT_ALG_LASSO']["bReconstruct_lasso"]:
        try:
            temp = np.array([lasso_algo(params, normalized_path[0]), lasso_algo(params, normalized_path[1])])
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif params['RCT_ALG_BFGS']["bReconstruct_bfgs"]:
        try:
            temp = np.array([bfgs_algo(params, normalized_path[0]), bfgs_algo(params, normalized_path[1])])
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif params['RCT_ALG_NN']["bReconstruct_NN"]:
        nnObj = CNeuralNetwork(params)
        try:
            temp = nnObj.nn_inference(normalized_path)
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    return reconstructed_paths
