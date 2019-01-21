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
# NUMBER_POINTS = 50e3
NUMBER_POINTS = 64e3

iterable_params = []
# noise_levels = [100, 80, 50, 40, 30, 20, 10, 7, 5, 3, 2, 1, 0.5]
noise_levels = [40, 30, 20, 10, 7, 5, 3, 2, 1, 0.5]
lasso_learning_rates = [0.01]
sampling_ratios = [0.2]
path_lengths = [256]

for lasso_learning_rate in lasso_learning_rates:
    for sampling_ratio in sampling_ratios:
        for path_length in path_lengths:
            for noise_level in noise_levels:
                realization = math.ceil(NUMBER_POINTS / path_length)
                iterable_params.append((sampling_ratio, path_length, lasso_learning_rate, realization, noise_level))

iterable_length = range(0, len(iterable_params), 1)

# Business logic for input arguments to main function
numberOfArgument = len(sys.argv)
if numberOfArgument == 3:
    flag = sys.argv[1]  # first argument should be the filenames identifier
    configFile = sys.argv[2]
else:
    print('Filenames identifier must be given and config file to be used or too many arguments given')
    sys.exit(0)


# The main processing function
def CSNN_proc(index):
    sampling_ratio_in = iterable_params[index][0]
    path_length_in = iterable_params[index][1]
    lasso_learning_rate_in = iterable_params[index][2]
    realization_in = iterable_params[index][3]
    noise_level_in = iterable_params[index][4]

    framework_model = cFramework()
    framework_model.update_framework(['CSNN_proc', configFile])
    framework_model.local_struct["realization"] = realization_in
    framework_model.local_struct["acquisition_time_sec"] = path_length_in
    framework_model.local_struct["noise_level_meter"] = [noise_level_in]
    framework_model.local_struct["filename"] = \
        "%s_%.f_%.3f_%.3f_%02d" % (flag, path_length_in, sampling_ratio_in, lasso_learning_rate_in, index)
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
