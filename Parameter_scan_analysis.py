# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : Parameter_scan_analysis.py
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
import _pickle as pickle
import matplotlib.pyplot as plt


workingDir      = os.getcwd()
resultsPath     = workingDir + '\\Results\\'
result_files    = os.listdir(resultsPath)
flag = 'PStest2'

parameter_scan_files  = [file for file in result_files if flag in file]
number_of_points      = len(parameter_scan_files)
sampling_ratios       = np.zeros(number_of_points)
path_lengths          = np.zeros(number_of_points)
SNR                   = np.zeros(number_of_points)
MSE                   = np.zeros(number_of_points)

for i in range(number_of_points) :
    file = parameter_scan_files[i]
    with open(resultsPath+file, 'rb') as txt_file_read :
        temp               = pickle.load(txt_file_read)
        sampling_ratios[i] = temp['RCT_ALG_LASSO']['sampling_ratio']
        path_lengths[i]    = temp['acquisition_length']
        SNR[i]             = np.mean(temp["RESULTS"]['reconstructed_db_latlon']['Lasso']) #mean can be dangerous if more than 1 noise level
        MSE[i]             = np.mean(temp["RESULTS"]['MSE_latlon']['Lasso'])

plt.scatter(sampling_ratios, path_lengths, c=SNR,cmap='rainbow_r')
plt.colorbar()
plt.show()        
plt.scatter(sampling_ratios, path_lengths, c=MSE,cmap='rainbow_r')
plt.colorbar()
plt.show()        