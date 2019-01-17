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


def split_array(arr, l, bOCS=True):
    size = int(arr.shape[0])
    overlap = int(l / 2)
    last = int((size / overlap) * overlap)
    clipped_array = arr[:last]

    blocks = []
    if bOCS:
        for i in range(int((size / overlap)) - 1):
            temp = clipped_array[i * overlap:(i * overlap + l)]
            blocks.append(temp)
    else:
        for j in range(size // l):
            temp = clipped_array[j * l: (j + 1) * l]
            blocks.append(temp)

    return np.array(blocks)


def merge_arrays(lis, l, bOCS=True):
    if bOCS:
        temp = lis[0, :int(3 * l / 4)]
        size = lis.shape[0]

        for i in range(1, size - 1):
            temp = np.concatenate((temp, lis[i, int(l / 4):int(3 * l / 4)]))

        if size > 1:
            temp = np.concatenate((temp, lis[-1, int(l / 4):]))
    else:
        temp = np.array([])
        for block in lis:
            temp = np.concatenate((temp, block))

    return temp


def lassoOps(params, adaptive, normalized_path, noise_dist):
    if params["bSplit"]:
        lat_blocks = split_array(normalized_path[0], params["block_length"], bOCS=params["bOCS"])
        lon_blocks = split_array(normalized_path[1], params["block_length"], bOCS=params["bOCS"])
        noise_blocks = split_array(noise_dist, params["block_length"], bOCS=params["bOCS"])
        num_blocks = len(noise_blocks)
    else:
        lat_blocks = normalized_path[0].reshape(1, len(normalized_path[0]))
        lon_blocks = normalized_path[1].reshape(1, len(normalized_path[1]))
        noise_blocks = noise_dist.reshape(1, len(noise_dist))
        num_blocks = 1

    temp_lat = []
    temp_lon = []
    temp_ratios = []

    for i in range(num_blocks):
        adaptiveObj = cAdaptiveSampling(params, adaptive, noise_blocks[i])
        lassoObj = cLasso(params)
        samples, final_sampling_ratio = adaptiveObj.adaptiveSample()
        print(samples)
        temp_lat.append(lassoObj.reconstructor(lat_blocks[i], samples))
        temp_lon.append(lassoObj.reconstructor(lon_blocks[i], samples))
        temp_ratios.append(final_sampling_ratio)

    final_lat = merge_arrays(np.array(temp_lat), params["block_length"], bOCS=params["bOCS"])
    final_lon = merge_arrays(np.array(temp_lon), params["block_length"], bOCS=params["bOCS"])
    final_sampling_ratio = np.mean(temp_ratios)
    temp = np.array([final_lat, final_lon])

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

    if (("ALG_LASSO" in algorithm) or ("ALG_BFGS" in algorithm)) and params[algorithm]["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params[algorithm], False, normalized_path, noise_dist)
        except ValueError as valerr:
            raise CFrameworkError(valerr.args[0]) from valerr

    elif "ALG_ADAPTIVE" in algorithm and params[algorithm]["bReconstruct"]:
        try:
            temp, final_sampling_ratio = lassoOps(params[algorithm], True, normalized_path, noise_dist)
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
