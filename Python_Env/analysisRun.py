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
from Helper_functions.proc_results import cProcessFile

# Logging
import logging

logger = logging.getLogger('BckTrk')

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"

workingDir = os.getcwd()
resultsPath = workingDir + direc_ident + 'Results' + direc_ident
result_files = os.listdir(resultsPath)


class CRunAnalysis:
    def __init__(self, user_input):
        # Business logic for input arguments to main function
        if len(user_input) == 2:
            self.flag = sys.argv[1]  # first argument should be the filenames identifier
        else:
            print('Filenames identifier must be given')
            sys.exit(0)

        self.parameter_scan_files = []

    def scan_files(self):
        for file in result_files:
            if self.flag in file:
                if '.csv' not in file:
                    self.parameter_scan_files.append(file)

    def analyseNoise(self):
        self.parameter_scan_files = np.sort(self.parameter_scan_files)
        number_of_points = len(self.parameter_scan_files)
        if number_of_points == 0:
            print('Wrong filenames identifier, no files found')
            sys.exit(0)

        with open(resultsPath + self.parameter_scan_files[0], 'rb') as txt_file_read:
            temp = pickle.load(txt_file_read)

        localStruct = temp
        noise_level_len = number_of_points
        numberOfRealizations = temp['realization']

        reconstructed_latlon_paths = temp["RESULTS"]['reconstructed_db_latlon']

        noise_levels = np.zeros(noise_level_len)
        MSE_r_latlon = {}
        final_sampling_ratio = {}

        for key in reconstructed_latlon_paths.keys():
            MSE_r_latlon[key] = np.zeros(len(noise_levels))
            final_sampling_ratio[key] = np.zeros((numberOfRealizations, noise_level_len))

        for i in range(number_of_points):
            file = self.parameter_scan_files[i]

            with open(resultsPath + file, 'rb') as txt_file_read:
                try:
                    temp = pickle.load(txt_file_read)
                except IOError:
                    print("Error loading file <%s>, skipping...") % txt_file_read
                else:
                    noise_levels[i] = temp['noise_level_meter'][0]  # script designed with only one value of noise here
                    for key in reconstructed_latlon_paths.keys():
                        MSE_r_latlon[key][i] = temp["RESULTS"]['MSE_latlon'][key]
                        final_sampling_ratio[key][:, i] = temp["RESULTS"]['final_sampling_ratio'][key][:, 0]

        # Extra parameters required based on what is being plotted
        localStruct["noise_level_meter"] = noise_levels

        # Results to be saved
        localStruct["RESULTS"]['final_sampling_ratio'] = final_sampling_ratio
        localStruct["RESULTS"]['MSE_latlon'] = MSE_r_latlon
        localStruct['RESULTS']['reconstructed_latlon_paths'] = reconstructed_latlon_paths
        localStruct['RESULTS']['paths_wm_org'] = 0
        localStruct['RESULTS']['paths_latlon_org'] = 0
        localStruct['RESULTS']['paths_wm_noisy'] = 0
        localStruct['RESULTS']['paths_latlon_noisy'] = 0
        localStruct['RESULTS']['transformed_paths'] = 0
        localStruct['RESULTS']['reconstructed_WM_paths'] = 0
        localStruct['RESULTS']['noise_vals'] = 0
        # What to plot
        localStruct["PLOT"]['bPlotMSE'] = True
        localStruct["PLOT"]['bPlotAvgSR'] = True
        return localStruct


if __name__ == "__main__":
    analyzeObj = CRunAnalysis(sys.argv)
    analyzeObj.scan_files()
    struct = analyzeObj.analyseNoise()

    data_obj = cProcessFile(struct)
    data_obj.plotAvgSR()
    data_obj.plotAccuracy()
    dummy = np.zeros(len(struct["noise_level_meter"]))
    data_obj.plot_MSE(dummy, dummy, dummy, dummy, struct["RESULTS"]['MSE_latlon'])
