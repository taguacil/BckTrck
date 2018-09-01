# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : Parameter_scan.py
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
import os
import json


workingDir      = os.getcwd()
paramPath       = workingDir + '\\'
local_struct    = json.load(open(paramPath + 'Parameter_files\\parameter_scan.json', 'r'))

sampling_ratios = np.array([0.05,0.07,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9])
path_lengths = np.arange(200,5000,100, dtype=int)

#sampling_ratios = np.arange(0.05,0.5,0.025)
#path_lengths = [200,300,400,500,600,700,800,900,1000]

for path_length in path_lengths :
    for sampling_ratio in sampling_ratios :
        local_struct["acquisition_time_sec"] = path_length
        local_struct["RCT_ALG_LASSO"]["sampling_ratio"] = sampling_ratio
        local_struct["filename"] = "PSrun1_%.f_%.3f"%(path_length, sampling_ratio)
        
        with open(paramPath + 'Parameter_files\\parameter_scan.json', 'w') as outfile:
            json.dump(local_struct, outfile)
            
        os.system("python CSNN_Framework.py Parameter_files\\parameter_scan.json")
        
        
        