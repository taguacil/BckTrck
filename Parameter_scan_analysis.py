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
from ast import literal_eval

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"

workingDir = os.getcwd()
resultsPath = workingDir + direc_ident + 'Results' + direc_ident
result_files = os.listdir(resultsPath)

# Business logic for input arguments to main function
numberOfArgument = len(sys.argv)
if numberOfArgument >= 2:
    flag = sys.argv[1]  # first argument should be the filenames identifier
else:
    print('Filenames identifier must be given')
    sys.exit(0)

parameter_scan_files = []
bcsv_generated = False
for file in result_files:
    if flag in file:
        if '.csv' not in file:
            parameter_scan_files.append(file)
        else:
            try:
                table = pd.read_csv(resultsPath + file)
            except FileNotFoundError:
                print("Wrong filename identifier or no file found, exiting...")
                sys.exit(0)
            else:
                bcsv_generated = True
                sampling_ratios_values = table.SamplingRatio.unique()
                path_lengths_values = table.PathLengths.unique()
                learning_rates_values = table.LearningRate.unique()
                noise_levels_values = table.Noise.unique()
                break

if bcsv_generated:
    if numberOfArgument == 3:
        sliceTuple_string = sys.argv[2]  # second argument should be a slice tuple
        sliceTuple = literal_eval(sliceTuple_string)
    else:
        print('Default slice to be plotted (0,0,0,0)')
        sliceTuple = (0, 0, 0, 0)  # sampling ratio, path lenghts, learning rate, noise levels

else:
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
    noise_levels_values = np.unique(noise_levels)

    arr = np.array(
        [sampling_ratios, path_lengths, learning_rates, noise_levels, l1_norm_lat, l1_norm_lon, SNR,
         MSE * 1e5]).transpose()
    table = pd.DataFrame(arr, columns=['SamplingRatio', 'PathLengths', 'LearningRate', 'Noise', 'L1Lat', 'L1Lon', 'SNR',
                                       'MSE'])

    print("Dataframe creation complete, dumping and starting plotting")
    table.to_csv(resultsPath + flag + "_debugTable" + '.csv', encoding='utf-8', index=False)
    sliceTuple = (0, 0, 0, 0)

sampling_ratios_slice = sampling_ratios_values[sliceTuple[0]]
path_lengths_slice = path_lengths_values[sliceTuple[1]]
learning_rates_slice = learning_rates_values[sliceTuple[2]]
noise_levels_slice = noise_levels_values[sliceTuple[3]]

fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

table[(table.LearningRate == learning_rates_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('SamplingRatio', 'PathLengths', c='MSE', colormap='rainbow_r', ax=axes[0, 0],
                 title='Lr <%.3f>, Noise <%.3f>' % (learning_rates_slice, noise_levels_slice), logy=True)

table[(table.PathLengths == path_lengths_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('SamplingRatio', 'LearningRate', c='MSE', colormap='rainbow_r', ax=axes[0, 1],
                 title='Pl <%d>, Noise <%.3f>' % (path_lengths_slice, noise_levels_slice), logy=True)

table[(table.LearningRate == learning_rates_slice) & (table.PathLengths == path_lengths_slice)]. \
    plot.scatter('SamplingRatio', 'Noise', c='MSE', colormap='rainbow_r', ax=axes[0, 2],
                 title='Pl <%d>, Lr <%.3f>' % (path_lengths_slice, learning_rates_slice), logy=True)

table[(table.SamplingRatio == sampling_ratios_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('PathLengths', 'LearningRate', c='MSE', colormap='rainbow_r', ax=axes[1, 0],
                 title='Sr <%.3f>, Noise <%.3f>' % (sampling_ratios_slice, noise_levels_slice), logy=True)

table[(table.LearningRate == learning_rates_slice) & (table.SamplingRatio == sampling_ratios_slice)]. \
    plot.scatter('PathLengths', 'Noise', c='MSE', colormap='rainbow_r', ax=axes[1, 1],
                 title='Sr <%.3f>, Lr <%.3f>' % (sampling_ratios_slice, learning_rates_slice), logy=True)

table[(table.SamplingRatio == sampling_ratios_slice) & (table.PathLengths == path_lengths_slice)]. \
    plot.scatter('LearningRate', 'Noise', c='MSE', colormap='rainbow_r', ax=axes[1, 2],
                 title='Lr <%.3f>, Noise <%.3f>' % (learning_rates_slice, noise_levels_slice), logy=True)

plt.show()

"""
fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

table[(table.LearningRate == learning_rates_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('SamplingRatio', 'PathLengths', c='SNR', colormap='rainbow_r', ax=axes[0, 0])

table[(table.PathLengths == path_lengths_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('SamplingRatio', 'LearningRate', c='SNR', colormap='rainbow_r', ax=axes[0, 1])

table[(table.LearningRate == learning_rates_slice) & (table.PathLengths == path_lengths_slice)]. \
    plot.scatter('SamplingRatio', 'Noise', c='SNR', colormap='rainbow_r', ax=axes[0, 2])

table[(table.SamplingRatio == sampling_ratios_slice) & (table.Noise == noise_levels_slice)]. \
    plot.scatter('PathLengths', 'LearningRate', c='SNR', colormap='rainbow_r', ax=axes[1, 0])

table[(table.LearningRate == learning_rates_slice) & (table.SamplingRatio == sampling_ratios_slice)]. \
    plot.scatter('PathLengths', 'Noise', c='SNR', colormap='rainbow_r', ax=axes[1, 1])

table[(table.SamplingRatio == sampling_ratios_slice) & (table.PathLengths == path_lengths_slice)]. \
    plot.scatter('LearningRate', 'Noise', c='SNR', colormap='rainbow_r', ax=axes[1, 2])

plt.show()
"""

table[(table.LearningRate == learning_rates_slice) & (table.SamplingRatio == sampling_ratios_slice)]. \
    plot.scatter('PathLengths', 'Noise', c='L1Lat', colormap='rainbow_r',
                 title='L1 norm for latitude', logy=True)
plt.show()

table[(table.LearningRate == learning_rates_slice) & (table.SamplingRatio == sampling_ratios_slice)]. \
    plot.scatter('PathLengths', 'Noise', c='L1Lon', colormap='rainbow_r',
                 title='L1 norm for longitude', logy=True)
plt.show()
