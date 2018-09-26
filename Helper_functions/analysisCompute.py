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


class CScanAnalysis:
    def __init__(self, user_input):
        # Business logic for input arguments to main function
        if len(user_input) == 2:
            self.flag = sys.argv[1]  # first argument should be the filenames identifier
        else:
            print('Filenames identifier must be given')
            sys.exit(0)

        self.parameter_scan_files = []

        self.table = pd.DataFrame()
        self.sampling_ratios_values = []
        self.path_lengths_values = []
        self.learning_rates_values = []
        self.noise_levels_values = []

    def scan_files(self):
        bcsv_generated = False

        for file in result_files:
            if self.flag in file:
                if '.csv' not in file:
                    self.parameter_scan_files.append(file)
                else:
                    try:
                        table = pd.read_csv(resultsPath + file)
                    except FileNotFoundError:
                        print("Wrong filename identifier or no file found, exiting...")
                        sys.exit(0)
                    else:
                        bcsv_generated = True
                        self.sampling_ratios_values = table.SamplingRatio.unique()
                        self.path_lengths_values = table.PathLengths.unique()
                        self.learning_rates_values = table.LearningRate.unique()
                        self.noise_levels_values = table.Noise.unique()
                        self.table = table
                        break
        return bcsv_generated

    def generate_table(self):
        number_of_points = len(self.parameter_scan_files)
        if number_of_points == 0:
            print('Wrong filenames identifier, no files found')
            sys.exit(0)

        with open(resultsPath + self.parameter_scan_files[0], 'rb') as txt_file_read:
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
            file = self.parameter_scan_files[i]

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

        self.sampling_ratios_values = np.unique(sampling_ratios)
        self.path_lengths_values = np.unique(path_lengths)
        self.learning_rates_values = np.unique(learning_rates)
        self.noise_levels_values = np.unique(noise_levels)

        arr = np.array(
            [sampling_ratios, path_lengths, learning_rates, noise_levels, l1_norm_lat, l1_norm_lon, SNR,
             MSE * 1e5]).transpose()
        table = pd.DataFrame(arr,
                             columns=['SamplingRatio', 'PathLengths', 'LearningRate', 'Noise', 'L1Lat', 'L1Lon', 'SNR',
                                      'MSE'])

        print("Dataframe creation complete, dumping and starting plotting")
        table.to_csv(resultsPath + self.flag + "_debugTable" + '.csv', encoding='utf-8', index=False)
        self.table = table

    def analyze(self):
        bcsv_generated = self.scan_files()
        if not bcsv_generated:
            self.generate_table()

    def get_panda_table(self):
        return self.table

    def get_slices_values(self):
        return self.sampling_ratios_values, self.path_lengths_values, \
               self.learning_rates_values, self.noise_levels_values
