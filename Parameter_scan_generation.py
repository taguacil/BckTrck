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
from concurrent import futures
import numpy as np

import sys

## User-defined library import
from CSNN_Framework import cFramework

WORKERS=5
sampling_ratios = np.array([0.05,0.07,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1])
path_lengths = [200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000]

#sampling_ratios = np.arange(0.05,0.5,0.025)
#path_lengths = [200,300,400,500,600,700,800,900,1000]

## Business logic for input arguments to main function 
numberOfArgument =  len(sys.argv)  
if numberOfArgument == 2 :
    flag = sys.argv[1] #first argument should be the filenames identifier
else :
    print ('Filenames identifier must be given')
    sys.exit(0)

## The main processing function
def CSNN_proc(path_length, sampling_ratio): 
        
        framework_model = cFramework()
        framework_model.update_framework( [0, 'Parameter_files\\parameter_scan_def.json'] )
        framework_model.local_struct["acquisition_time_sec"] = path_length
        framework_model.local_struct["RCT_ALG_LASSO"]["sampling_ratio"] = sampling_ratio
        framework_model.local_struct["filename"] = "%s_%.f_%.3f"%(flag, path_length, sampling_ratio)
        framework_model.mainComputation(framework_model.local_struct)


## The main loop
for path_length in path_lengths :
    for sampling_ratio in sampling_ratios :
        CSNN_proc(path_length, sampling_ratio)        