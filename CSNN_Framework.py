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
import platform

## User-defined library import
from Helper_functions.proc_results import process_data 
from Helper_functions.transforms import transforms 
from Navigation.Random_walker import random_2d_path_generator
import Navigation.Coordinates as cord
from Navigation.AWGN import noise_generator
from Reconstruction_algorithms.Master_reconstruction import reconstructor, identify_algorithms
from Helper_functions.csv_interpreter import munge_csv

if platform.system() == "Windows" :
    direc_ident = "\\"
else :
    direc_ident = "/"
        

def update(dictionary, updateDict):
    for k, v in updateDict.items():
        if isinstance(v, collections.Mapping):
            dictionary[k] = update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary
    
class cFramework:
    ## Constructor
    def __init__(self):
         
        ## Parameters / Config files handling
        workingDir      = os.getcwd()
        self.paramPath       = workingDir + direc_ident
        self.local_struct    = json.load(open(self.paramPath + 'Parameter_files'+ direc_ident + 'default_config.json', 'r'))
        
        try :
            os.stat (self.paramPath+'Logs'+ direc_ident)
        except : 
            os.mkdir(self.paramPath+'Logs'+ direc_ident)
                
        try :
            os.stat (self.paramPath+'Results'+ direc_ident)
        except :
            os.mkdir(self.paramPath+'Results'+ direc_ident)
             
        # create logger with 'spam_application'
        self.logger = logging.getLogger('BckTrk')
        self.logger.setLevel(logging.INFO)
        
        # create file handler which logs even debug messages
        now = datetime.datetime.now()
        self.fh = logging.FileHandler(self.paramPath+'Logs'+ direc_ident + 'BckTrk_Log_' + now.strftime("%Y-%m-%d")+'.log')
        self.fh.setLevel(logging.INFO)
        
        self.local_struct['currentTime'] = now
        self.local_struct['workingDir']  = workingDir
        
        # create console handler with same log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        
        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)


    def update_framework(self, arguments):
        numberOfArgument =  len(arguments)  
        if numberOfArgument == 1 :
            self.logger.info ('Default config will be used')
            
        elif numberOfArgument == 2 :
            parameters_filename = arguments[1] #first argument should be the parameters filename
            updated_config = json.load(open( self.paramPath + parameters_filename, 'r')) #JSON loads a dictionary written in a JSON file as a python dictionary
            self.local_struct = update (self.local_struct,updated_config)
        else :
            self.logger.error ('Invalid number of arguments %d' %numberOfArgument)
            self.exit_framework()
        
    ## Define exit function
    
    # TODO Since an exception can occur in one of the sub modules, this function will never be called 
    # Doubling logs bug will happen, solution 1 make all function return a value to main
    # pass the exit function somehow to all submodules 
    
    def exit_framework(self):
        self.fh.close()
        self.ch.close()
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)
        sys.exit(0)

    ## Main function definition 
    def mainComputation(self, local_struct) :
        
        ##Variables initialization
        use_random_seed                     = local_struct['bUse_random_seed']
        random_seed                         = local_struct['random_seed']
        numberOfRealizations                = local_struct['realization']
        noise_level                         = local_struct['noise_level_meter']
        noise_level_len                     = len(noise_level)
        
        use_csv_data                        = local_struct['CSV_DATA']['bUse_csv_data']
        csv_path                            = local_struct['CSV_DATA']['csv_path']
        path_length                         = local_struct['CSV_DATA']['path_length']
        
		
        ##Set seed
        if use_random_seed :
            np.random.seed(random_seed)
        
        if use_csv_data :
            
            local_struct['acquisition_length'] = path_length
            paths_latlon_real, latlon_accuracy, latlon_interval = munge_csv(csv_path, path_length)
            local_struct['realization'] = latlon_accuracy.shape[-1]
            numberOfRealizations = local_struct['realization']
            
            paths_wm_real = np.zeros((2,path_length,numberOfRealizations))
            transformed_real_paths = np.zeros((2,path_length,numberOfRealizations))
            
            reconstructed_real_latlon_paths          = {}
            reconstructed_real_WM_paths              = {}
            
            reconstruction_algorithms           = identify_algorithms(local_struct)
            for algorithm in reconstruction_algorithms:
                reconstructed_real_latlon_paths[algorithm]  = np.zeros((2,path_length,numberOfRealizations))
                reconstructed_real_WM_paths[algorithm]      = np.zeros((2,path_length,numberOfRealizations)) 
                
            for realization in range(numberOfRealizations):
                transformed_real_paths[:,:,realization] = transforms(local_struct,paths_latlon_real[:,:,realization])
                paths_wm_real[:,:,realization] = cord.generate_WM_array(paths_latlon_real[:,:,realization])
                if local_struct['bReconstruct'] :
                    temp = reconstructor(local_struct, paths_latlon_real[:,:,realization])
                    for algorithm in reconstruction_algorithms :
                        reconstructed_real_latlon_paths[algorithm][:,:,realization] = temp[algorithm][:,:]
                        reconstructed_real_WM_paths[algorithm][:,:,realization] = cord.generate_WM_array(temp[algorithm][:,:])
            
            local_struct['RESULTS']['paths_wm_real']                     = paths_wm_real
            local_struct['RESULTS']['paths_latlon_real']                 = paths_latlon_real
            local_struct['RESULTS']['transformed_real_paths']            = transformed_real_paths
            local_struct['RESULTS']['reconstructed_real_latlon_paths']       = reconstructed_real_latlon_paths
            local_struct['RESULTS']['reconstructed_real_WM_paths']           = reconstructed_real_WM_paths    
            
            
            
        else : 
        ##Iterate over the total number of realizations
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
                
            self.logger.info ('Starting simulation with <%d> realizations and <%d> path length',numberOfRealizations,acquisition_length)
            for realization in range(numberOfRealizations):
                ##Generate random data
                self.logger.debug ('Generating random data for realization <%d>',realization)
                acquisition_length                  = local_struct['gps_freq_Hz']*local_struct['acquisition_time_sec']
            
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
        
        self.logger.info ('Generating results and plotting')
        process_data(local_struct)
        self.exit_framework()
    
## Main function definition  MUST BE at the END OF FILE    
if __name__ == "__main__" :
    ## Business logic for input arguments to main function 
    framework_model = cFramework()
    framework_model.update_framework( sys.argv )
    framework_model.mainComputation(framework_model.local_struct)
