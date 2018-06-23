# %-*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : CSNN_Framework.py
 -----------------------------------------------------------------------------

   Description :

   This file contains the main function which is the core of the simulation.
   More details here  
   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   25-Apr-2018  1.0      Rami      File created
 =============================================================================
 
"""
## Python library import
import numpy as np
import sys
import os
import json
import datetime
import collections
import logging

## User-defined library import
from Helper_functions.proc_results import process_data 
from Helper_functions.transforms import transforms 
from Navigation.Random_walker import random_2d_path_generator
import Navigation.Coordinates as cord
from Navigation.AWGN import noise_generator
from Reconstruction_algorithms.Master_reconstruction import reconstructor, identify_algorithms


## Parameters / Config files handling
workingDir      = os.getcwd()
paramPath       = workingDir + '\\'
local_struct    = json.load(open(paramPath + 'Parameter_files\\default_config.json', 'r'))

try :
    os.stat (workingDir+'\\Logs\\')
except : 
    os.mkdir(workingDir+'\\Logs\\')
        
try :
    os.stat (workingDir+'\\Results\\')
except :
    os.mkdir(workingDir+'\\Results\\')
     
# create logger with 'spam_application'
logger = logging.getLogger('BckTrk')
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
now = datetime.datetime.now()
fh = logging.FileHandler(workingDir + '\\Logs\\' + 'BckTrk_Log_' + now.strftime("%Y-%m-%d")+'.log')
fh.setLevel(logging.INFO)

local_struct['currentTime'] = now
local_struct['workingDir']  = workingDir

# create console handler with same log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

## Define exit function

# TODO Since an exception can occur in one of the sub modules, this function will never be called 
# Doubling logs bug will happen, solution 1 make all function return a value to main
# pass the exit function somehow to all submodules 

def exit_framework():
     
    fh.close()
    ch.close()
    logger.removeHandler(fh)
    logger.removeHandler(ch)
    sys.exit(0)

def update(dictionary, updateDict):
    for k, v in updateDict.items():
        if isinstance(v, collections.Mapping):
            dictionary[k] = update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary

## Business logic for input arguments to main function 
numberOfArgument =  len(sys.argv)  
if numberOfArgument == 1 :
    logger.info ('Default config will be used')
    
elif numberOfArgument == 2 :
    parameters_filename = sys.argv[1] #first argument should be the parameters filename
    updated_config = json.load(open( paramPath + parameters_filename, 'r')) #JSON loads a dictionary written in a JSON file as a python dictionary
    local_struct = update (local_struct,updated_config)
else :
    logger.error ('Invalid number of arguments %d' %numberOfArgument)
    exit_framework()
    

## Main function definition 
def main() :
    ##Variables initialization
    use_random_seed                     = local_struct['bUse_random_seed']
    random_seed                         = local_struct['random_seed']
    numberOfRealizations                = local_struct['realization']
    noise_level                         = local_struct['noise_level_meter']
    noise_level_len                     = len(noise_level)
    
    acquisition_length                  = local_struct['gps_freq_Hz']*local_struct['acquisition_time_sec']
    local_struct['acquisition_length']  = acquisition_length
    paths_wm_org                        = np.zeros((2,acquisition_length,numberOfRealizations))
    paths_latlon_org                    = np.zeros((2,acquisition_length,numberOfRealizations))
    paths_wm_noisy                      = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
    paths_latlon_noisy                  = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
    noise_vals                          = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
    transformed_paths                   = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
    reconstructed_latlon_paths          = {}
    reconstructed_WM_paths              = {}
    
    reconstruction_algorithms           = identify_algorithms(local_struct)
    for algorithm in reconstruction_algorithms:
        reconstructed_latlon_paths[algorithm]  = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
        reconstructed_WM_paths[algorithm]      = np.zeros((2,acquisition_length,numberOfRealizations,noise_level_len))
    
    ##Set seed
    if use_random_seed :
        np.random.seed(random_seed)
        
    ##Iterate over the total number of realizations        
    for realization in range(numberOfRealizations):
        ##Generate random data
        logger.info ('Generating random data for realization <%d>',realization)
        
        (paths_wm_org[:,:,realization],paths_latlon_org[:,:,realization]) = random_2d_path_generator(local_struct)
        for lvl in range(noise_level_len):
            (paths_wm_noisy[:,:,realization,lvl],paths_latlon_noisy[:,:,realization,lvl],noise_vals[:,:,realization,lvl]) = noise_generator(local_struct,paths_wm_org[:,:,realization],noise_level[lvl])
            transformed_paths[:,:,realization,lvl]=transforms(local_struct,paths_latlon_noisy[:,:,realization,lvl])
            if local_struct['bReconstruct'] :
                temp = reconstructor(local_struct, paths_latlon_noisy[:,:,realization,lvl])
                for algorithm in reconstruction_algorithms :
                    reconstructed_latlon_paths[algorithm][:,:,realization,lvl] = temp[algorithm][:,:]
                    reconstructed_WM_paths[algorithm][:,:,realization,lvl] = cord.generate_WM_array(temp[algorithm][:,:])
    #Store data in local struct
    local_struct['RESULTS']['paths_wm_org']                     = paths_wm_org
    local_struct['RESULTS']['paths_latlon_org']                 = paths_latlon_org
    local_struct['RESULTS']['paths_wm_noisy']                   = paths_wm_noisy
    local_struct['RESULTS']['paths_latlon_noisy']               = paths_latlon_noisy
    local_struct['RESULTS']['transformed_paths']                = transformed_paths
    local_struct['RESULTS']['reconstructed_latlon_paths']       = reconstructed_latlon_paths
    local_struct['RESULTS']['reconstructed_WM_paths']           = reconstructed_WM_paths
    
    logger.info ('Generating results and plotting')
    process_data(local_struct)
    exit_framework()
    
## Main function definition  MUST BE at the END OF FILE    
if __name__ == "__main__" :
    main()
    
