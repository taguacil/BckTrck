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
from .Lasso_reconstruction import cLasso
from .BFGS_reconstruction import cBFGS
from .Adaptive_sampling import cAdaptiveSampling
from Helper_functions.framework_error import CFrameworkError
from Helper_functions.framework_error import CErrorTypes

# Logging
import logging

logger = logging.getLogger('BckTrk')


# Helper functions
def identify_algorithms(params):
    temp = []
    for key in params.keys():
        if "RCT_ALG" in key:
            if params[key]["bReconstruct"]:
                temp.append(key)

    return temp


def lassoOps(params, adaptive, normalized_path, noise_dist):
    adaptiveObj = cAdaptiveSampling(params, adaptive, noise_dist)
    lassoObj = cLasso(params)
    samples, final_sampling_ratio = adaptiveObj.adaptiveSample()
    normalized_lat = lassoObj.reconstructor(normalized_path[0], samples)
    normalized_lon = lassoObj.reconstructor(normalized_path[1], samples)
    temp = np.array([normalized_lat, normalized_lon])
    return temp, final_sampling_ratio


def bfgsOps(params, adaptive, normalized_path, noise_dist):
    adaptiveObj = cAdaptiveSampling(params, adaptive, noise_dist)
    bfgsObj = cBFGS(params)
    samples, final_sampling_ratio = adaptiveObj.adaptiveSample()
    normalized_lat = bfgsObj.bfgs_run(normalized_path[0], samples)
    normalized_lon = bfgsObj.bfgs_run(normalized_path[1], samples)
    temp = np.array([normalized_lat, normalized_lon])
    return temp, final_sampling_ratio


# The main processing function
def reconstructor(params, path, algorithm, noise_dist):
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')

    mean = np.repeat(np.expand_dims(np.mean(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    var = np.repeat(np.expand_dims(np.var(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    normalized_path = (path - mean) / np.sqrt(var)

    if algorithm == "RCT_ALG_LASSO" and params['RCT_ALG_LASSO']["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params['RCT_ALG_LASSO'], False, normalized_path, noise_dist)
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    if algorithm == "RCT_ALG_ADAPTIVE_LASSO" and params['RCT_ALG_ADAPTIVE_LASSO']["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params['RCT_ALG_ADAPTIVE_LASSO'], True, normalized_path, noise_dist)
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif algorithm == "RCT_ALG_BFGS" and params['RCT_ALG_BFGS']["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params['RCT_ALG_BFGS'], False, normalized_path, noise_dist)
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif algorithm == "RCT_ALG_ADAPTIVE_BFGS" and params['RCT_ALG_ADAPTIVE_BFGS']["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params['RCT_ALG_ADAPTIVE_BFGS'], True, normalized_path, noise_dist)
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif "NN" in algorithm and params[algorithm]["bReconstruct"]:
        logger.debug('Entering NN reconstruction')
        try:
            logger.debug('Beginning inference')
            nn_name = algorithm + "Obj"
            temp = np.zeros((2, params['acquisition_length']))
            temp[0], temp[1] = params[nn_name].nn_inference(normalized_path)
            final_sampling_ratio = params[algorithm]["sampling_ratio"]
        except (ValueError, KeyError) as valerr:
            logger.debug('NN value error with message <%s>', valerr.args[0])
            errdict = {"file": __file__, "message": valerr.args[0], "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict) from valerr

    reconstructed_paths = np.sqrt(var) * temp + mean
    return reconstructed_paths, final_sampling_ratio
