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


def split_array(path, num_blocks, blockLength, middleBuffer, bufferLength):
    logger.debug('Splitting array starts')
    blocks = np.zeros((num_blocks, blockLength))
    # padd tails
    tail_beg = np.repeat(path[0], bufferLength)
    tail_end = np.repeat(path[-1], bufferLength+blockLength)
    path_padded = np.concatenate((tail_beg, path, tail_end))
    for m in range(1, num_blocks + 1):
        debut = (m - 1) * middleBuffer
        fin = debut + blockLength
        blocks[m - 1, :] = path_padded[debut: fin]
    return blocks


def merge_arrays(path_blocks, num_blocks, blockLength, bufferLength, acquisitionLength):
    logger.debug('Merging arrays starts')
    temp = path_blocks[0, bufferLength:(blockLength - bufferLength)]
    for i in range(1, num_blocks-1):
        temp = np.concatenate((temp, path_blocks[i, bufferLength:(blockLength - bufferLength)]))
    newSize = len(temp)
    if newSize < acquisitionLength:
        # Padd the end because of finite length
        temp = np.concatenate((temp, path_blocks[-1, bufferLength:bufferLength+(acquisitionLength - newSize)]))

    return temp


def DevOps(algorithm, params, normalized_path, noise_dist):
    logger.debug('Algorithm devOps starts')
    params_alg = params[algorithm]
    blockLength = int(params_alg["block_length"])
    bufferLength = int(params_alg["buffer_length"])
    acquisitionLength = int(params["acquisition_length"])
    adaptive = False

    middleBuffer = (blockLength - 2 * bufferLength)
    if middleBuffer <= 0:
        logger.debug("Overlapping buffer cannot be > block")
        errdict = {"file": __file__, "message": "Overlapping buffer cannot be > block", "errorType": CErrorTypes.value}
        raise ValueError(errdict)
    num_blocks = int(np.ceil(acquisitionLength / middleBuffer))  # ceil to cover all cases below bufferlength 64

    if "ADAPTIVE" in algorithm:
        logger.debug("Adaptive sampling enabaled")
        adaptive = True
    if "LASSO" in algorithm:
        devObj = cLasso(params_alg)
    elif "BFGS" in algorithm:
        devObj = cBFGS(params_alg)
    elif "NN" in algorithm:
        logger.debug('Entering NN reconstruction')
        logger.debug('Beginning inference')
        nn_name = algorithm + "Obj"
        devObj = params[nn_name]

    if num_blocks == 0:
        logger.debug("Number of blocks must be > 0")
        errdict = {"file": __file__, "message": "Number of blocks must be > 0", "errorType": CErrorTypes.value}
        raise ValueError(errdict)
    elif num_blocks == 1:
        lat_blocks = normalized_path[0].reshape(1, acquisitionLength)
        lon_blocks = normalized_path[1].reshape(1, acquisitionLength)
        noise_blocks = noise_dist.reshape(1, acquisitionLength)
    else:
        lat_blocks = split_array(normalized_path[0], num_blocks, blockLength, middleBuffer, bufferLength)
        lon_blocks = split_array(normalized_path[1], num_blocks, blockLength, middleBuffer, bufferLength)
        noise_blocks = split_array(noise_dist, num_blocks, blockLength, middleBuffer, bufferLength)

    temp_lat = np.zeros((num_blocks, blockLength))
    temp_lon = np.zeros((num_blocks, blockLength))
    temp_ratios = np.zeros(num_blocks)

    mean_lat = np.repeat(np.expand_dims(np.mean(lat_blocks, axis=1), axis=1), blockLength, axis=1)
    var_lat = np.repeat(np.expand_dims(np.var(lat_blocks, axis=1), axis=1), blockLength, axis=1)
    normalized_lat_blocks = (lat_blocks - mean_lat) / np.sqrt(var_lat)

    mean_lon = np.repeat(np.expand_dims(np.mean(lon_blocks, axis=1), axis=1), blockLength, axis=1)
    var_lon = np.repeat(np.expand_dims(np.var(lon_blocks, axis=1), axis=1), blockLength, axis=1)
    normalized_lon_blocks = (lon_blocks - mean_lon) / np.sqrt(var_lon)

    for i in range(num_blocks):
        adaptiveObj = cAdaptiveSampling(params_alg, adaptive, noise_blocks[i])
        samples, temp_ratios[i] = adaptiveObj.adaptiveSample()
        temp_lat[i, :] = devObj.run(normalized_lat_blocks[i], samples, blockLength)
        temp_lon[i, :] = devObj.run(normalized_lon_blocks[i], samples, blockLength)

    temp_lat = np.sqrt(var_lat) * temp_lat + mean_lat
    temp_lon = np.sqrt(var_lon) * temp_lon + mean_lon

    final_lat = merge_arrays(temp_lat, num_blocks, blockLength, bufferLength, acquisitionLength)
    final_lon = merge_arrays(temp_lon, num_blocks, blockLength, bufferLength, acquisitionLength)
    final_sr = np.mean(temp_ratios)
    localtemp = np.array([final_lat, final_lon])

    return localtemp, final_sr


# The main processing function
def reconstructor(params, path, algorithm, noise_dist):
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')

    if params[algorithm]["bReconstruct"]:
        try:
            reconstructed_paths, final_sampling_ratio = DevOps(algorithm, params, path, noise_dist)
        except (ValueError, KeyError) as valerr:
            logger.debug('<%s> value error with message <%s>', algorithm, valerr.args[0])
            errdict = {"file": __file__, "message": valerr.args[0], "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict) from valerr

    return reconstructed_paths, final_sampling_ratio