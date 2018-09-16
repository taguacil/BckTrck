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
else:
    print('Filenames identifier must be given')
    sys.exit(0)

parameter_scan_files = [file for file in result_files if flag in file]
number_of_points = len(parameter_scan_files)
if number_of_points == 0:
    print('Wrong filenames identifier, no files found')
    sys.exit(0)

with open(resultsPath + parameter_scan_files[0], 'rb') as txt_file_read:
    temp = pickle.load(txt_file_read)
    noise_levels = np.array(temp['noise_level_meter'])

len_noise_levels = noise_levels.shape[0]

sampling_ratios = np.zeros((number_of_points, len_noise_levels))
path_lengths = np.zeros((number_of_points, len_noise_levels))
learning_rates = np.zeros((number_of_points, len_noise_levels))

MSE = np.zeros((number_of_points, len_noise_levels))
SNR = np.zeros((number_of_points, len_noise_levels))

l1_norm_lat = np.zeros((number_of_points, len_noise_levels))
l1_norm_lon = np.zeros((number_of_points, len_noise_levels))

for i in range(number_of_points):
    file = parameter_scan_files[i]

    with open(resultsPath + file, 'rb') as txt_file_read:
        try:
            temp = pickle.load(txt_file_read)
        except IOError:
            print("Error loading file <%s>, skipping...") % txt_file_read
        else:
            sampling_ratios[i, :] = temp['RCT_ALG_LASSO']['sampling_ratio'] * np.ones(len_noise_levels)
            path_lengths[i, :] = temp['acquisition_length'] * np.ones(len_noise_levels)
            learning_rates[i, :] = temp['RCT_ALG_LASSO']['lasso_learning_rate'] * np.ones(len_noise_levels)

            # prepare the l1 norm
            if temp['TRANSFORM']['bDctTransform']:
                l1_norm_lat[i, :] = \
                    np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][0], ord=1, axis=0), axis=0)
                l1_norm_lon[i, :] = \
                    np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][1], ord=1, axis=0), axis=0)

            SNR[i, :] = temp["RESULTS"]['reconstructed_db_latlon']['Lasso']
            MSE[i, :] = temp["RESULTS"]['MSE_latlon']['Lasso']
            print("File <%d> processed out of <%d>") % (i+1, number_of_points)

sampling_ratios = np.unique(sampling_ratios)
path_lengths = np.unique(path_lengths)
learning_rates = np.unique(learning_rates)

len_sampling_ratios = sampling_ratios.shape[0]
len_path_lengths = path_lengths.shape[0]
len_learning_rates = learning_rates.shape[0]

MSE = MSE.reshape((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))
SNR = SNR.reshape((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))
l1_norm_lat = l1_norm_lat.reshape((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))
l1_norm_lon = l1_norm_lon.reshape((len_path_lengths, len_sampling_ratios, len_learning_rates, len_noise_levels))

print("Reshape complete, plotting")

plt.figure()

plt.subplot(2, 3, 1)
plt.pcolormesh(MSE[:, :, 0, 0] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Sampling ratio')
plt.ylabel('Path Length')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.pcolormesh(MSE[:, 0, :, 0] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Learning rate')
plt.ylabel('Path Length')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.pcolormesh(MSE[:, 0, 0, :] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Noise level')
plt.ylabel('Path Length')
plt.colorbar()

plt.subplot(2, 3, 4)
plt.pcolormesh(MSE[0, :, :, 0] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.yticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Learning rate')
plt.ylabel('Sampling ratio')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.pcolormesh(MSE[0, :, 0, :] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Noise level')
plt.ylabel('Sampling ratio')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.pcolormesh(MSE[0, 0, :, :] * 1e5, cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.title('MSE of reconstructed (1e-5)')
plt.xlabel('Noise level')
plt.ylabel('Learning rate')
plt.colorbar()

plt.tight_layout()
plt.show()

fig = plt.figure()

plt.subplot(2, 3, 1)
plt.pcolormesh(SNR[:, :, 0, 0], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.xlabel('Sampling ratio')
plt.ylabel('Path Length')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.pcolormesh(SNR[:, 0, :, 0], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.xlabel('Learning rate')
plt.ylabel('Path Length')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.pcolormesh(SNR[:, 0, 0, :], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.xlabel('Noise level')
plt.ylabel('Path Length')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.subplot(2, 3, 4)
plt.pcolormesh(SNR[0, :, :, 0], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.yticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.xlabel('Learning rate')
plt.ylabel('Sampling ratio')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.pcolormesh(SNR[0, :, 0, :], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_sampling_ratios - 0.5), len_sampling_ratios), sampling_ratios)
plt.xlabel('Noise level')
plt.ylabel('Sampling ratio')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.subplot(2, 3, 6)
plt.pcolormesh(SNR[0, 0, :, :], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_learning_rates - 0.5), len_learning_rates), learning_rates)
plt.xlabel('Noise level')
plt.ylabel('Learning rate')
plt.title('SNR [dB] of MSE ratios')
plt.colorbar()

plt.tight_layout()
plt.show()

plt.pcolormesh(l1_norm_lat[:, 0, 0, :], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.xlabel('Noise level')
plt.ylabel('Path Length')
plt.title('L1 norm of lat')
plt.colorbar()
plt.show()

plt.pcolormesh(l1_norm_lon[:, 0, 0, :], cmap='rainbow_r')
plt.xticks(np.linspace(0.5, (len_noise_levels - 0.5), len_noise_levels), noise_levels)
plt.yticks(np.linspace(0.5, (len_path_lengths - 0.5), len_path_lengths), path_lengths)
plt.xlabel('Noise level')
plt.ylabel('Path Length')
plt.title('L1 norm of lon')
plt.colorbar()
plt.show()
