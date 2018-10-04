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
from Helper_functions.framework_error import CErrorTypes

# Logging
import logging

logger = logging.getLogger('BckTrk')

# Helper functions
def identify_algorithms(params):
    temp = []
    if params['RCT_ALG_LASSO']["bReconstruct_lasso"]:
        temp.append("RCT_ALG_LASSO")
    if params['RCT_ALG_BFGS']["bReconstruct_bfgs"]:
        temp.append("RCT_ALG_BFGS")
    if params['RCT_ALG_NN']["bReconstruct_NN"]:
        temp.append("RCT_ALG_NN")
    return temp


# The main processing function
def reconstructor(params, path, algorithm):
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')

    mean = np.repeat(np.expand_dims(np.mean(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    var = np.repeat(np.expand_dims(np.var(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    normalized_path = (path - mean) / np.sqrt(var)

    if algorithm == "RCT_ALG_LASSO" and params['RCT_ALG_LASSO']["bReconstruct_lasso"]:
        try:
            temp = np.array([lasso_algo(params, normalized_path[0]), lasso_algo(params, normalized_path[1])])
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif algorithm == "RCT_ALG_BFGS" and params['RCT_ALG_BFGS']["bReconstruct_bfgs"]:
        try:
            temp = np.array([bfgs_algo(params, normalized_path[0]), bfgs_algo(params, normalized_path[1])])
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif algorithm == "RCT_ALG_NN" and params['RCT_ALG_NN']["bReconstruct_NN"]:
        logger.debug('Entering NN reconstruction')
        try:
            logger.debug('Beginning inference')
            temp = np.zeros((2, params['acquisition_length']))
            temp[0], temp[1] =params['nnObj'].nn_inference(normalized_path)
            reconstructed_paths = np.sqrt(var) * temp + mean
        except ValueError as valerr:
            logger.debug('NN value error with message <%s>',valerr.args[0])
            errdict = {"file": __file__, "message" : valerr.args[0], "errorType": CErrorTypes.value }
            raise CFrameworkError(errdict) from valerr

    return reconstructed_paths
