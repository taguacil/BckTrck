# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : Parameter_scan_analysis.py
 -----------------------------------------------------------------------------

   Description :

   This file contains the analysis of the generated scan files,
   generic enough to handle missing files
   
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
import pandas as pd

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
    noise_levels_values = np.array(temp['noise_level_meter'])

len_noise_levels = noise_levels_values.shape[0]

sampling_ratios = np.zeros((number_of_points * len_noise_levels))
path_lengths = np.zeros((number_of_points * len_noise_levels))
learning_rates = np.zeros((number_of_points * len_noise_levels))
noise_levels = np.zeros((number_of_points * len_noise_levels))

MSE = np.zeros((number_of_points * len_noise_levels))
SNR = np.zeros((number_of_points * len_noise_levels))

l1_norm_lat = np.zeros((number_of_points * len_noise_levels))
l1_norm_lon = np.zeros((number_of_points * len_noise_levels))

for i in range(number_of_points):
    file = parameter_scan_files[i]

    with open(resultsPath + file, 'rb') as txt_file_read:
        try:
            temp = pickle.load(txt_file_read)
        except IOError:
            print("Error loading file <%s>, skipping...") % txt_file_read
        else:
            start = i * len_noise_levels
            end = (i + 1) * len_noise_levels
            sampling_ratios[start:end] = temp['RCT_ALG_LASSO']['sampling_ratio'] * np.ones(len_noise_levels)
            path_lengths[start:end] = temp['acquisition_length'] * np.ones(len_noise_levels)
            learning_rates[start:end] = temp['RCT_ALG_LASSO']['lasso_learning_rate'] * np.ones(len_noise_levels)
            noise_levels[start:end] = np.array(temp['noise_level_meter'])

            # prepare the l1 norm
            if temp['TRANSFORM']['bDctTransform']:
                l1_norm_lat[start:end] = \
                    np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][0], ord=1, axis=0), axis=0)
                l1_norm_lon[start:end] = \
                    np.mean(np.linalg.norm(temp["RESULTS"]['transformed_paths'][1], ord=1, axis=0), axis=0)

            SNR[start:end] = temp["RESULTS"]['reconstructed_db_latlon']['Lasso']
            MSE[start:end] = temp["RESULTS"]['MSE_latlon']['Lasso']
            print("File <%d> processed out of <%d>" % (i + 1, number_of_points))

sampling_ratios_values = np.unique(sampling_ratios)
path_lengths_values = np.unique(path_lengths)
learning_rates_values = np.unique(learning_rates)

arr = np.array(
    [sampling_ratios, path_lengths, learning_rates, noise_levels, l1_norm_lat, l1_norm_lon, SNR, MSE * 1e5]).transpose()
table = pd.DataFrame(arr, columns=['Sr', 'Pl', 'Lr', 'N', 'L1lat', 'L1lon', 'SNR', 'MSE'])

print("Dataframe creation complete, dumping and starting plotting")
table.to_csv(resultsPath + "debugTable" + '.csv', encoding='utf-8', index=False)

fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

table[(table.Lr == learning_rates_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Sr', 'Pl', c='MSE', colormap='rainbow_r', ax=axes[0, 0])

table[(table.Pl == path_lengths_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Sr', 'Lr', c='MSE', colormap='rainbow_r', ax=axes[0, 1])

table[(table.Lr == learning_rates_values[0]) & (table.Pl == path_lengths_values[0])]. \
    plot.scatter('Sr', 'N', c='MSE', colormap='rainbow_r', ax=axes[0, 2])

table[(table.Sr == sampling_ratios_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Pl', 'Lr', c='MSE', colormap='rainbow_r', ax=axes[1, 0])

table[(table.Lr == learning_rates_values[0]) & (table.Sr == sampling_ratios_values[0])]. \
    plot.scatter('Pl', 'N', c='MSE', colormap='rainbow_r', ax=axes[1, 1])

table[(table.Sr == sampling_ratios_values[0]) & (table.Pl == path_lengths_values[0])]. \
    plot.scatter('Lr', 'N', c='MSE', colormap='rainbow_r', ax=axes[1, 2])

plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

table[(table.Lr == learning_rates_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Sr', 'Pl', c='SNR', colormap='rainbow_r', ax=axes[0, 0])

table[(table.Pl == path_lengths_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Sr', 'Lr', c='SNR', colormap='rainbow_r', ax=axes[0, 1])

table[(table.Lr == learning_rates_values[0]) & (table.Pl == path_lengths_values[0])]. \
    plot.scatter('Sr', 'N', c='SNR', colormap='rainbow_r', ax=axes[0, 2])

table[(table.Sr == sampling_ratios_values[0]) & (table.N == noise_levels_values[0])]. \
    plot.scatter('Pl', 'Lr', c='SNR', colormap='rainbow_r', ax=axes[1, 0])

table[(table.Lr == learning_rates_values[0]) & (table.Sr == sampling_ratios_values[0])]. \
    plot.scatter('Pl', 'N', c='SNR', colormap='rainbow_r', ax=axes[1, 1])

table[(table.Sr == sampling_ratios_values[0]) & (table.Pl == path_lengths_values[0])]. \
    plot.scatter('Lr', 'N', c='SNR', colormap='rainbow_r', ax=axes[1, 2])

plt.show()

table[(table.Lr == learning_rates_values[0]) & (table.Sr == sampling_ratios_values[0])]. \
    plot.scatter('Pl', 'N', c='L1lat', colormap='rainbow_r')
plt.title('L1 norm of lat')
plt.show()

table[(table.Lr == learning_rates_values[0]) & (table.Sr == sampling_ratios_values[0])]. \
    plot.scatter('Pl', 'N', c='L1lon', colormap='rainbow_r')
plt.title('L1 norm of lon')
plt.show()
