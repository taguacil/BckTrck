"""
 =============================================================================
 Title       : Neural network model design and training
 Project     : Simulation environment for BckTrk app
 File        : NN_train.py
 -----------------------------------------------------------------------------

   Description :

    This file is responsible for training a designed network and generating the mode for inference

   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   27-Sep-2018  1.0      Taimir      File created
 =============================================================================

"""
# Python library import
import numpy as np
import keras
import os
import platform

# User-defined library import
from Helper_functions.framework_error import CFrameworkError
from Helper_functions.framework_error import CErrorTypes

# Logging
import logging

logger = logging.getLogger('BckTrk')

if platform.system() == "Windows":
    direc_ident = "\\"
else:
    direc_ident = "/"

workingDir = os.getcwd()
resultsPath = workingDir + direc_ident + 'NeuralNetworks' + direc_ident + 'Models' + direc_ident


class CNeuralNetwork:
    # Constructor
    def __init__(self, struct, algorithm):

        self.messageSummary_dict = {}
        self.identifier = ""  # To describe which model the message belongs to

        if struct[algorithm]["sampling_ratio"] > 1:
            logger.debug("Sampling_ratio larger than 1")
            errdict = {"file": __file__, "message": "Sampling_ratio larger than 1", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.m_acquisition_length = struct['acquisition_length']
        self.alpha = struct[algorithm]["alpha"]
        if struct['bTrainNetwork']:
            self.m_model_lat = keras.Sequential()
            self.m_model_lon = keras.Sequential()
        else:
            modelname_lat = resultsPath + struct[algorithm]["modelname"]
            modelname_lon = resultsPath + struct[algorithm]["modelname"]
            try:
                self.load_models(modelname_lat, modelname_lon)
            except FileNotFoundError:
                message = 'Model <%s> in directory <%s>not found!' % (struct[algorithm]["modelname"], resultsPath)
                logger.debug(message)
                errdict = {"file": __file__, "message": message, "errorType": CErrorTypes.value}
                raise CFrameworkError(errdict)

        self.number_of_samples = int(struct[algorithm]["sampling_ratio"] * struct["acquisition_length"])
        self.realizations = struct['realization']

        if self.number_of_samples <= 0:
            logger.debug("Number of samples cannot be 0 or negative")
            errdict = {"file": __file__, "message": "Invalid number of samples", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.noiseLevel_len = len(struct['noise_level_meter'])
        if self.noiseLevel_len <= 0:
            logger.debug("Noise array cannot be empty")
            errdict = {"file": __file__, "message": "Noise array is empty", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

    def activation_fun(self, x):
        return keras.activations.relu(x, alpha=self.alpha)

    def design_nn(self):
        # For lat
        self.m_model_lat.add(
            keras.layers.Dense(self.m_acquisition_length, activation=self.activation_fun,
                               input_shape=(self.number_of_samples,)))
        self.m_model_lat.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lat.add(keras.layers.Dropout(0.1))
        self.m_model_lat.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.m_model_lat.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lat.add(keras.layers.Dropout(0.1))
        self.m_model_lat.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.m_model_lat.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lat.add(keras.layers.Dense(self.m_acquisition_length, activation="linear"))

        self.m_model_lat.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"]
        )

        # For lon
        self.m_model_lon.add(
            keras.layers.Dense(self.m_acquisition_length, activation=self.activation_fun,
                               input_shape=(self.number_of_samples,)))
        self.m_model_lon.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lon.add(keras.layers.Dropout(0.1))
        self.m_model_lon.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.m_model_lon.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lon.add(keras.layers.Dropout(0.1))
        self.m_model_lon.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.m_model_lon.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.m_model_lon.add(keras.layers.Dense(self.m_acquisition_length, activation="linear"))

        self.m_model_lon.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"]
        )

        self.m_model_lat.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"]
        )

    def train_nn(self, org_latlon, input_latlon_noisy):
        # First we normalize
        samples = np.sort(np.random.choice(self.m_acquisition_length, self.number_of_samples, replace=False))
        downsampled_latlon = input_latlon_noisy[:, samples, :, :]

        paths_latlon_org, paths_latlon_noisy = self.normalize_path_training(org_latlon, downsampled_latlon)

        # W reshape to have a matrix for lat and lon respectively
        # [PL;Realizations;Noises] --> [Realizations*Noises;PL]
        totalrealizations = self.realizations * self.noiseLevel_len

        lat_y = np.transpose(np.reshape(paths_latlon_org[0, :, :, :],
                                        (self.m_acquisition_length, totalrealizations)))
        lon_y = np.transpose(np.reshape(paths_latlon_org[1, :, :, :],
                                        (self.m_acquisition_length, totalrealizations)))
        lat_X = np.transpose(np.reshape(paths_latlon_noisy[0, :, :, :],
                                        (self.number_of_samples, totalrealizations)))
        lon_X = np.transpose(np.reshape(paths_latlon_noisy[1, :, :, :],
                                        (self.number_of_samples, totalrealizations)))

        # Randomly pick 80% training and 20% validation from totalrealizations
        validateIndices = np.random.choice(totalrealizations, int(np.floor(0.2 * totalrealizations)), replace=False)
        trainIndices = np.array([n for n in range(totalrealizations) if n not in validateIndices])

        # Train the models
        results_lat = self.m_model_lat.fit(
            lat_X[trainIndices, :], lat_y[trainIndices, :],
            epochs=1000,
            batch_size=32,
            validation_data=(lat_X[validateIndices, :], lat_y[validateIndices, :]),
            verbose=2
        )

        results_lon = self.m_model_lon.fit(
            lon_X[trainIndices, :], lon_y[trainIndices, :],
            epochs=1000,
            batch_size=32,
            validation_data=(lon_X[validateIndices, :], lon_y[validateIndices, :]),
            verbose=2
        )

    def dump_nn_summary(self):
        # Dumps summary to debugger output and to log file saved in logs
        self.identifier = "latitude"
        self.m_model_lat.summary(print_fn=self.custom_print)
        self.identifier = "longitude"
        self.m_model_lon.summary(print_fn=self.custom_print)
        return True

    def nn_inference(self, path_latlon_noisy):
        # Perform inference on given path 
        samples = np.sort(np.random.choice(self.m_acquisition_length, self.number_of_samples, replace=False))
        path_latlon_noisy_dnw = path_latlon_noisy[:,samples].transpose()
        
        path_lat = np.array([path_latlon_noisy_dnw[:, 0]])
        path_lon = np.array([path_latlon_noisy_dnw[:, 1]])
        
        # Check if vector length is the same as input layer is implicitly done by ValueError
        path_lat_reconst = self.m_model_lat.predict(path_lat)
        path_lon_reconst = self.m_model_lon.predict(path_lon)
        
        return path_lat_reconst, path_lon_reconst

    def save_models(self, modelname_lat, modelname_lon):
        # serialize model to JSON
        model_lat_json = self.m_model_lat.to_json()
        model_lon_json = self.m_model_lon.to_json()
        with open(modelname_lat + "_lat.json", "w") as json_file:
            json_file.write(model_lat_json)
        with open(modelname_lon + "_lon.json", "w") as json_file:
            json_file.write(model_lon_json)
        # serialize weights to HDF5
        self.m_model_lat.save_weights(modelname_lat + "_lat.h5")
        self.m_model_lon.save_weights(modelname_lon + "_lon.h5")

    def load_models(self, modelname_lat, modelname_lon):
        # Loads both models from unique names from directory NeuralNetworks/Models
        modelpath_lat_json = modelname_lat + "_lat.json"
        modelpath_lon_json = modelname_lat + "_lon.json"
        modelpath_lat_h5 = modelname_lat + "_lat.h5"
        modelpath_lon_h5 = modelname_lat + "_lon.h5"
        # load json and create model
        json_file = open(modelpath_lat_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.m_model_lat = keras.models.model_from_json(loaded_model_json, 
                                                        custom_objects={'activation_fun': self.activation_fun})
        json_file = open(modelpath_lon_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.m_model_lon = keras.models.model_from_json(loaded_model_json, 
                                                        custom_objects={'activation_fun': self.activation_fun})
        # load weights into new model
        self.m_model_lat.load_weights(modelpath_lat_h5)
        self.m_model_lon.load_weights(modelpath_lon_h5)

    def normalize_path_training(self, paths_latlon_org, paths_latlon_noisy):
        # Normalizing both downsampled and original data
        mean = np.repeat(np.expand_dims(np.mean(paths_latlon_org, axis=1), axis=1), self.m_acquisition_length, axis=1)
        var = np.repeat(np.expand_dims(np.var(paths_latlon_org, axis=1), axis=1), self.m_acquisition_length, axis=1)
        paths_latlon_org_norm = (paths_latlon_org - mean) / np.sqrt(var)

        mean = np.repeat(np.expand_dims(np.mean(paths_latlon_noisy, axis=1), axis=1), self.number_of_samples, axis=1)
        var = np.repeat(np.expand_dims(np.var(paths_latlon_noisy, axis=1), axis=1), self.number_of_samples, axis=1)
        paths_latlon_noisy_norm = (paths_latlon_noisy - mean) / np.sqrt(var)

        return paths_latlon_org_norm, paths_latlon_noisy_norm

    def custom_print(self, message):
        # Will be called by model.summary() for every line in the summary
        logger.info(message)
        if not self.identifier:
            logger.debug("No NN model was chosen")
            errdict = {"file": __file__, "message": "No NN model was chosen", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)
        else:
            if self.identifier in self.messageSummary_dict.keys():
                    self.messageSummary_dict[self.identifier] += message
            else:
                self.messageSummary_dict[self.identifier] = message

