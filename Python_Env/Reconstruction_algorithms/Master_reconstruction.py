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


def split_array(arr, l, bOCS=False):
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


def merge_arrays(lis, l, bOCS=False):
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


def DevOps(algorithm, params, normalized_path, noise_dist):
    params_alg = params[algorithm]
    blockLength = params_alg["block_length"]
    acquisitionLength = params["acquisition_length"]
    useOCS = params_alg["bOCS"]
    adaptive = False

    num_blocks = np.floor(acquisitionLength / blockLength)  # must be a multiple for now
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
        lat_blocks = split_array(normalized_path[0], blockLength, bOCS=useOCS)
        lon_blocks = split_array(normalized_path[1], blockLength, bOCS=useOCS)
        noise_blocks = split_array(noise_dist, blockLength, bOCS=useOCS)

    temp_lat = np.zeros(blockLength, num_blocks)
    temp_lon = np.zeros(blockLength, num_blocks)
    temp_ratios = np.zeros(num_blocks)

    for i in range(num_blocks):
        adaptiveObj = cAdaptiveSampling(params_alg, adaptive, noise_blocks[i])
        samples, temp_ratios[i] = adaptiveObj.adaptiveSample()
        temp_lat[:, i] = devObj.run(lat_blocks[i], samples, blockLength)
        temp_lon[:, i] = devObj.run(lon_blocks[i], samples, blockLength)

    final_lat = merge_arrays(temp_lat, blockLength, bOCS=useOCS)
    final_lon = merge_arrays(temp_lon, blockLength, bOCS=useOCS)
    final_sampling_ratio = np.mean(temp_ratios)
    temp = np.array([final_lat, final_lon])

    return temp, final_sampling_ratio


# The main processing function
def reconstructor(params, path, algorithm, noise_dist):
    logger.debug('Returns dictionary of different path reconstructions using different algorithms')

    mean = np.repeat(np.expand_dims(np.mean(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    var = np.repeat(np.expand_dims(np.var(path, axis=1), axis=1), params['acquisition_length'], axis=1)
    normalized_path = (path - mean) / np.sqrt(var)

    if params[algorithm]["bReconstruct"]:
        try:
            temp, final_sampling_ratio = DevOps(algorithm, params, normalized_path, noise_dist)
        except (ValueError, KeyError) as valerr:
            logger.debug('<%s> value error with message <%s>', algorithm, valerr.args[0])
            errdict = {"file": __file__, "message": valerr.args[0], "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict) from valerr

    reconstructed_paths = np.sqrt(var) * temp + mean
    return reconstructed_paths, final_sampling_ratio
