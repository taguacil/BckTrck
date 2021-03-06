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
   12-Sep-2018  1.1      Taimir    Extension to support variable learning rate
 =============================================================================
 
"""
# Python library import
from concurrent import futures
import sys
import platform
import math
import json
import os

# User-defined library import

from CSNN_Framework import cFramework

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"

workingDir = os.getcwd()
resultsPath = workingDir + direc_ident + 'Logs' + direc_ident

WORKERS = 16
NUMBER_POINTS = 50e3

iterable_params = []
noise_levels = [0.0005, 5, 10, 20, 40, 100, 200]
lasso_learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1]
sampling_ratios = [0.05, 0.07, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
path_lengths = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
                2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]

for lasso_learning_rate in lasso_learning_rates:
    for sampling_ratio in sampling_ratios:
        for path_length in path_lengths:
            realization = math.ceil(NUMBER_POINTS / path_length)
            iterable_params.append((sampling_ratio, path_length, lasso_learning_rate, realization))

iterable_length = range(0, len(iterable_params), 1)

# Business logic for input arguments to main function
numberOfArgument = len(sys.argv)
if numberOfArgument == 2:
    flag = sys.argv[1]  # first argument should be the filenames identifier
else:
    print('Filenames identifier must be given or too many arguments given')
    sys.exit(0)


# The main processing function
def CSNN_proc(index):
    sampling_ratio_in = iterable_params[index][0]
    path_length_in = iterable_params[index][1]
    lasso_learning_rate_in = iterable_params[index][2]
    realization_in = iterable_params[index][3]

    framework_model = cFramework()
    framework_model.update_framework(['CSNN_proc', 'Parameter_files' + direc_ident + 'parameter_scan_def.json'])
    framework_model.local_struct["realization"] = realization_in
    framework_model.local_struct["acquisition_time_sec"] = path_length_in
    framework_model.local_struct["RCT_ALG_LASSO"]["sampling_ratio"] = sampling_ratio_in
    framework_model.local_struct["RCT_ALG_LASSO"]["lasso_learning_rate"] = lasso_learning_rate_in

    framework_model.local_struct["TRANSFORM"]["bDctTransform"] = True  # To save the transformed path
    framework_model.local_struct["noise_level_meter"] = noise_levels
    framework_model.local_struct["filename"] = \
        "%s_%.f_%.3f_%.3f" % (flag, path_length_in, sampling_ratio_in, lasso_learning_rate_in)
    reterrorDict = framework_model.mainComputation(framework_model.local_struct)
    return reterrorDict, sampling_ratio_in, path_length_in, lasso_learning_rate_in, realization_in


# The main loop
if __name__ == '__main__':
    with futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        returnVals = executor.map(CSNN_proc, iterable_length)
        logfile = {}
        passednumber = 0
        for retVal in returnVals:
            if retVal[0]["bNoErrors"]:
                passednumber += 1
            else:
                print("Generation failed for tuple (%.3f, %d, %.3f, %.3f)"
                      % (retVal[1], retVal[2], retVal[3], retVal[4]))
                logfile[str((retVal[1], retVal[2], retVal[3], retVal[4]))] = retVal[0]

        print("Map done, closing pool and terminating with <%d> out of <%d> passed generations"
              % (passednumber, len(iterable_params)))  # if there is a crash, it will not appear in the stats

        filename = resultsPath + 'BckTrk_scan_exception_' + flag + '.txt'

        with open(filename, "w") as data_file:
            json.dump(logfile, data_file, indent=4, sort_keys=True)
