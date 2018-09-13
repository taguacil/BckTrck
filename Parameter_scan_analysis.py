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
# Python library import
import numpy as np
import os
import sys
import _pickle as pickle
import matplotlib.pyplot as plt
import platform

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"

workingDir = os.getcwd()
resultsPath = workingDir + direc_ident + 'Results' + direc_ident
result_files = os.listdir(resultsPath)

# Business logic for input arguments to main function
numberOfArgument = len(sys.argv)
if numberOfArgument == 2:
    flag = sys.argv[1]  # first argument should be the filenames identifier
    with open(resultsPath + flag + '.txt', 'rb') as txt_file_read:
        temp_struct = pickle.load(txt_file_read)
else:
    print('Filenames identifier must be given')
    sys.exit(0)

parameter_scan_files = [file for file in result_files if flag in file]
number_of_points = len(parameter_scan_files)
if number_of_points == 0:
    print('Wrong filenames identifier, no files found')
    sys.exit(0)

sampling_ratios = np.array(temp_struct['sr'])
path_lengths = np.array(temp_struct['pl'])
learning_rates = np.array(temp_struct['lr'])
noise_levels = np.array(temp_struct['nl'])

len_sampling_ratios = sampling_ratios.shape[0]
len_path_lengths = path_lengths.shape[0]
len_learning_rates = learning_rates.shape[0]
len_noise_levels = noise_levels.shape[0]

MSE = np.zeros((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))
SNR = np.zeros((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))

l1_norm_lat = np.zeros((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))
l1_norm_lon = np.zeros((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))

for i in range(number_of_points):
    file = parameter_scan_files[i]

    with open(resultsPath + file, 'rb') as txt_file_read:
        temp = pickle.load(txt_file_read)
        sampling_ratio = temp['RCT_ALG_LASSO']['sampling_ratio']
        path_length = temp['acquisition_length']
        learning_rate = temp['RCT_ALG_LASSO']['lasso_learning_rate']

        index_sr = np.where(sampling_ratios == sampling_ratio)[0][0]
        index_pl = np.where(path_lengths == path_length)[0][0]
        index_lr = np.where(learning_rates == learning_rate)[0][0]

        noise_levels = temp['noise_level_meter']  # array

        # prepare the l1 norm
        if temp['TRANSFORM']['bDctTransform']:
            l1_norm_lat[index_pl, index_sr, index_lr, :] = \
                np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][0], ord=1, axis=0), axis=0)
            l1_norm_lon[index_pl, index_sr, index_lr, :] = \
                np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][1], ord=1, axis=0), axis=0)

        SNR[index_pl, index_sr, index_lr, :] = temp["RESULTS"]['reconstructed_db_latlon']['Lasso']
        MSE[index_pl, index_sr, index_lr, :] = temp["RESULTS"]['MSE_latlon']['Lasso']

# NOT implemented
plt.scatter(sampling_ratios, path_lengths, c=SNR, cmap='rainbow_r')
plt.colorbar()
plt.title('SNR [dB] of MSE ratios')
plt.xlabel('Sampling ratio')
plt.ylabel('Total number of samples')
plt.show()

plt.scatter(sampling_ratios, path_lengths, c=MSE, cmap='rainbow_r')
plt.colorbar()
plt.title('MSE of reconstructed coordinates')
plt.xlabel('Sampling ratio')
plt.ylabel('Total number of samples')
plt.show()
