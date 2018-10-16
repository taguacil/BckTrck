# %-*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : CSNN_Framework.py
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
import sys
import os
import json
import datetime
import collections
import logging
import platform

# User-defined library import
from Helper_functions.proc_results import process_data, get_pickle_file
from Helper_functions.transforms import transforms
from Navigation.Random_walker import random_2d_path_generator
import Navigation.Coordinates as cord
from Navigation.AWGN import noise_generator
from Reconstruction_algorithms.Master_reconstruction import reconstructor, identify_algorithms
from Helper_functions.csv_interpreter import munge_csv
from Helper_functions.framework_error import CFrameworkError
from Helper_functions.framework_error import CErrorTypes

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"


def update(dictionary, updateDict):
    for k, v in updateDict.items():
        if isinstance(v, collections.Mapping):
            dictionary[k] = update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


class cFramework:
    # Constructor
    def __init__(self):

        # Parameters / Config files handling
        workingDir = os.getcwd()
        self.paramPath = workingDir + direc_ident
        self.local_struct = json.load(
            open(self.paramPath + 'Parameter_files' + direc_ident + 'default_config.json', 'r'))

        try:
            os.stat(self.paramPath + 'Logs' + direc_ident)
        except:
            os.mkdir(self.paramPath + 'Logs' + direc_ident)

        try:
            os.stat(self.paramPath + 'Results' + direc_ident)
        except:
            os.mkdir(self.paramPath + 'Results' + direc_ident)

        # create logger with 'spam_application'
        self.logger = logging.getLogger('BckTrk')
        REALIZATION_log = 15
        logLevel = REALIZATION_log  # can be logging.INFO or DEBUG
        self.logger.setLevel(logLevel)

        # create file handler which logs even debug messages
        now = datetime.datetime.now()
        self.fh = logging.FileHandler(
            self.paramPath + 'Logs' + direc_ident + 'BckTrk_Log_' + now.strftime("%Y-%m-%d") + '.log')
        self.fh.setLevel(logLevel)

        self.local_struct['currentTime'] = now
        self.local_struct['workingDir'] = workingDir

        # create console handler with same log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logLevel)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

        self.frameworkError_list = {"bNoErrors": True}
        self.frameworklog_list = {"bNologs": True}

    def update_framework(self, arguments):
        numberOfArgument = len(arguments)
        if numberOfArgument == 1:
            self.logger.info('Default config will be used')

        elif numberOfArgument == 2:
            parameters_filename = arguments[1]  # first argument should be the parameters filename
            updated_config = json.load(open(self.paramPath + parameters_filename,
                                            'r'))  # loads a dictionary written in a JSON file as a python dictionary
            self.local_struct = update(self.local_struct, updated_config)
        else:
            self.logger.error('Invalid number of arguments %d' % numberOfArgument)
            self.exit_framework()

    # Define exit function

    # TODO Since an exception can occur in one of the sub modules, this function will never be called 
    # Doubling logs bug will happen, solution 1 make all function return a value to main
    # pass the exit function somehow to all submodules 

    def exit_framework(self):
        self.fh.close()
        self.ch.close()
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)

    # Main function definition
    def mainComputation(self, local_struct):

        # Variables initialization
        use_random_seed = local_struct['bUse_random_seed']
        random_seed = local_struct['random_seed']
        numberOfRealizations = local_struct['realization']
        noise_level_len = len(local_struct['noise_level_meter'])

        use_csv_data = local_struct['CSV_DATA']['bUse_csv_data']
        csv_path = local_struct['CSV_DATA']['csv_path']
        path_length = local_struct['CSV_DATA']['path_length']

        # Set seed
        if use_random_seed:
            np.random.seed(random_seed)

        # Iterate over the total number of realizations
        if use_csv_data:
            local_struct['noise_level_meter'] = [0]
            noise_level_len = 1

            acquisition_length = path_length
            paths_latlon_org, latlon_accuracy, latlon_interval = munge_csv(csv_path, path_length)
            local_struct['realization'] = latlon_accuracy.shape[-1]
            numberOfRealizations = local_struct['realization']
            paths_latlon_org = paths_latlon_org.reshape((2, path_length, numberOfRealizations, noise_level_len))
        else:
            acquisition_length = local_struct['gps_freq_Hz'] * local_struct['acquisition_time_sec']
            paths_latlon_org = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))

        local_struct['acquisition_length'] = acquisition_length

        paths_wm_org = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))
        paths_wm_noisy = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))
        paths_latlon_noisy = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))
        noise_vals = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))
        transformed_paths = np.zeros((2, acquisition_length, numberOfRealizations, noise_level_len))

        reconstructed_latlon_paths = {}
        reconstructed_WM_paths = {}
        bNN_initialized = {}

        reconstruction_algorithms = identify_algorithms(local_struct)
        for algorithm in reconstruction_algorithms:
            reconstructed_latlon_paths[algorithm] = np.zeros(
                (2, acquisition_length, numberOfRealizations, noise_level_len))
            reconstructed_WM_paths[algorithm] = np.zeros(
                (2, acquisition_length, numberOfRealizations, noise_level_len))
            bNN_initialized[algorithm] = False

        if local_struct['bTrainNetwork'] and local_struct['Train_NN']['bUseGeneratedData']:
            self.logger.info('Using generated data')
            try:
                dataFileName = 'TrainingSet'
                filename = self.paramPath + 'NeuralNetworks' + direc_ident + dataFileName + '.txt'
                try:
                    loadedStruct = get_pickle_file(filename)
                    # Assign the paths to local variables
                    # TODO some other checks on some parameters can be done
                    paths_latlon_org = loadedStruct['RESULTS']['paths_latlon_org']
                    paths_latlon_noisy = loadedStruct['RESULTS']['paths_latlon_noisy']
                except FileNotFoundError as filerr:
                    self.logger.debug("Training set not found")
                    errdict = {"file": __file__, "message": filerr.args[0],
                               "errorType": CErrorTypes.range}
                    raise CFrameworkError(errdict)
            except CFrameworkError as frameErr:
                self.errorAnalyzer(frameErr, "load training set")

        else:
            self.logger.info('Starting simulation with <%d> realizations and <%d> path length', numberOfRealizations,
                             acquisition_length)
            for lvl in range(noise_level_len):
                for realization in range(numberOfRealizations):
                    # Generate random data
                    self.logger.log(15, 'Generating random data for realization <%d>', realization)
                    if not use_csv_data:
                        (paths_wm_org[:, :, realization, lvl], paths_latlon_org[:, :, realization, lvl]) = \
                            random_2d_path_generator(local_struct)
                        # Generate noise for each realization
                        (paths_wm_noisy[:, :, realization, lvl], paths_latlon_noisy[:, :, realization, lvl],
                         noise_vals[:, :, realization, lvl]) = \
                            noise_generator(local_struct, paths_wm_org[:, :, realization, lvl],
                                            local_struct['noise_level_meter'][lvl])
                    else:
                        paths_wm_org[:, :, realization, lvl] = cord.generate_WM_array(
                            paths_latlon_org[:, :, realization, lvl])
                        paths_latlon_noisy[:, :, realization, lvl] = paths_latlon_org[:, :, realization, lvl]
                        paths_wm_noisy[:, :, realization, lvl] = paths_wm_org[:, :, realization, lvl]

                    # Apply transforms
                    if not local_struct['bTrainNetwork']:
                        transformed_paths[:, :, realization, lvl] = \
                            transforms(local_struct, paths_latlon_noisy[:, :, realization, lvl])

                        # Apply reconstruction algorithms
                        if local_struct['bReconstruct']:
                            for algorithm in reconstruction_algorithms:
                                if "NN" in algorithm and not bNN_initialized[algorithm]:
                                    from NeuralNetworks.NN import CNeuralNetwork
                                    nn_name = algorithm + "Obj"
                                    try:
                                        local_struct[nn_name] = CNeuralNetwork(local_struct, algorithm)
                                        bNN_initialized[algorithm] = True
                                    except CFrameworkError as frameErr:
                                        self.errorAnalyzer(frameErr, str((algorithm, lvl)))
                                try:
                                    temp = reconstructor(local_struct, paths_latlon_noisy[:, :, realization, lvl],
                                                         algorithm)
                                    reconstructed_latlon_paths[algorithm][:, :, realization, lvl] = temp
                                    try:
                                        reconstructed_WM_paths[algorithm][:, :, realization, lvl] = \
                                            cord.generate_WM_array(temp)
                                    except ValueError as valerr:
                                        self.logger.debug("Lat/Lon out of range in degrees")
                                        errdict = {"file": __file__, "message": valerr.args[0],
                                                   "errorType": CErrorTypes.range}
                                        raise CFrameworkError(errdict)
                                except CFrameworkError as frameErr:
                                    self.errorAnalyzer(frameErr, str((algorithm, lvl)))

        if local_struct['bTrainNetwork']:
            from NeuralNetworks.NN import CNeuralNetwork
            # Iterate over the total number of realizations to generate training set
            modelname_lat = self.paramPath + 'NeuralNetworks' + direc_ident + 'Models' + direc_ident \
                            + local_struct["Train_NN"]["modelname_lat"]
            modelname_lon = self.paramPath + 'NeuralNetworks' + direc_ident + 'Models' + direc_ident \
                            + local_struct["Train_NN"]["modelname_lon"]

            nnObj = CNeuralNetwork(local_struct, "Train_NN")
            nnObj.design_nn()
            results_lat, results_lon = nnObj.train_nn(paths_latlon_org, paths_latlon_noisy)                
            nnObj.save_models(modelname_lat, modelname_lon)

            # if nnObj.dump_nn_summary():
            #    self.logAnalyzer(nnObj.messageSummary_dict, modelname_lat)
            #    self.logAnalyzer(nnObj.messageSummary_dict, modelname_lon)
                
            if local_struct["Train_NN"]["bPlotTrainResults"]:
                nnObj.train_result_visu(results_lat, results_lon, local_struct["Train_NN"]["modelname_lat"],
                                        local_struct["Train_NN"]["modelname_lon"])

        # Store data in local struct
        local_struct['RESULTS']['paths_wm_org'] = paths_wm_org
        local_struct['RESULTS']['paths_latlon_org'] = paths_latlon_org
        local_struct['RESULTS']['paths_wm_noisy'] = paths_wm_noisy
        local_struct['RESULTS']['paths_latlon_noisy'] = paths_latlon_noisy
        local_struct['RESULTS']['transformed_paths'] = transformed_paths
        local_struct['RESULTS']['reconstructed_latlon_paths'] = reconstructed_latlon_paths
        local_struct['RESULTS']['reconstructed_WM_paths'] = reconstructed_WM_paths

        self.logger.debug('Generating results and plotting')
        try:
            process_data(local_struct)
        except CFrameworkError as frameErr:
            self.errorAnalyzer(frameErr, "process_data")

        self.exit_framework()
        return self.frameworkError_list

    def errorAnalyzer(self, frameErr, master_key):
        if self.frameworkError_list["bNoErrors"]:
            self.frameworkError_list["bNoErrors"] = False

        if master_key in self.frameworkError_list.keys():
            if frameErr.callermessage in self.frameworkError_list[master_key].keys():
                self.frameworkError_list[master_key][frameErr.callermessage] += 1
            else:
                self.frameworkError_list[master_key][frameErr.callermessage] = 1
        else:
            self.frameworkError_list[master_key] = {frameErr.callermessage: 1}

    def logAnalyzer(self, message, master_key):
        # No need to append the message to the master key for now (message is a dict in this case)
        if self.frameworklog_list["bNologs"]:
            self.frameworklog_list["bNologs"] = False

        self.frameworklog_list[master_key] = message


# Main function definition  MUST BE at the END OF FILE
if __name__ == "__main__":
    # Business logic for input arguments to main function
    framework_model = cFramework()
    framework_model.update_framework(sys.argv)
    frameworkError_list = framework_model.mainComputation(framework_model.local_struct)
    filename = framework_model.paramPath + 'Logs' + direc_ident + 'BckTrk_exception_' + \
               framework_model.local_struct["currentTime"].strftime("%Y-%m-%d") + '.json'

    with open(filename, "w") as data_file:
        json.dump(frameworkError_list, data_file, indent=4, sort_keys=True)

    if not framework_model.frameworklog_list["bNologs"]:
        filename = framework_model.paramPath + 'Logs' + direc_ident + 'BckTrk_logDump_' + \
                   framework_model.local_struct["currentTime"].strftime("%Y-%m-%d") + '.json'

        with open(filename, "w") as data_file:
            json.dump(framework_model.frameworklog_list, data_file, indent=4, sort_keys=True)
