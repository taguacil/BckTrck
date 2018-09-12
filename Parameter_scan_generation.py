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
import sys
import platform 

## User-defined library import
from CSNN_Framework import cFramework

if platform.system() == "Windows" :
    direc_ident = "\\"
else :
    direc_ident = "/"

WORKERS=16
sampling_ratios = [0.05,0.07,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
path_lengths    = [100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000]


iterable_params = []
for sampling_ratio in sampling_ratios :
    for path_length in path_lengths:
        iterable_params.append((sampling_ratio, path_length))
iterable_length = range(0,len(iterable_params),1)

## Business logic for input arguments to main function 
numberOfArgument =  len(sys.argv)  
if numberOfArgument == 2 :
    flag = sys.argv[1] #first argument should be the filenames identifier
else :
    print ('Filenames identifier must be given')
    sys.exit(0)

## The main processing function
def CSNN_proc(index): 
        sampling_ratio_in = iterable_params[index][0]
        path_length_in = iterable_params[index][1] 
        
        framework_model = cFramework()
        framework_model.update_framework( ['CSNN_proc', 'Parameter_files'+ direc_ident + 'parameter_scan_def.json'] )
        framework_model.local_struct["acquisition_time_sec"] = path_length_in
        framework_model.local_struct["RCT_ALG_LASSO"]["sampling_ratio"] = sampling_ratio_in
        framework_model.local_struct["filename"] = "%s_%.f_%.3f"%(flag, path_length_in, sampling_ratio_in)
        framework_model.mainComputation(framework_model.local_struct)

## The main loop
if __name__ == '__main__':
    Executor = futures.ProcessPoolExecutor(max_workers=WORKERS)
    Executor.map(CSNN_proc, iterable_length)
    print("Map done, closing pool and terminating...")