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
resultsPath = workingDir + direc_ident + 'Models' + direc_ident


class CNeuralNetwork:
    # Constructor
    def __init__(self, struct):

        if struct['RCT_ALG_NN']["sampling_ratio"] > 1:
            logger.debug("Sampling_ratio larger than 1")
            errdict = {"file": __file__, "message": "Sampling_ratio larger than 1", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.m_acquisition_length = struct['acquisition_length']
        if struct['bTrainNetwork']:
            self.m_model_lat = keras.Sequential()
            self.m_model_lon = keras.Sequential()
        else:
            modelname_lat = resultsPath + struct['RCT_ALG_NN']["modelname"] + "_lat.h5"
            modelname_lon = resultsPath + struct['RCT_ALG_NN']["modelname"] + "_lon.h5"
            try:
                self.load_models(modelname_lat, modelname_lon)
            except FileNotFoundError:
                message = 'Model <%s> in directory <%s>not found!' % (struct['RCT_ALG_NN']["modelname"], resultsPath)
                logger.debug(message)
                errdict = {"file": __file__, "message": message, "errorType": CErrorTypes.value}
                raise CFrameworkError(errdict)

        self.number_of_samples = int(struct['RCT_ALG_LASSO']["sampling_ratio"] * struct["acquisition_length"])
        self.realizations = struct['realization']

        if self.number_of_samples <= 0:
            logger.debug("Number of samples cannot be 0 or negative")
            errdict = {"file": __file__, "message": "Invalid number of samples", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.alpha = struct['RCT_ALG_NN']["alpha"]
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

    def train_nn(self, org_latlon, downsampled_latlon):
        # First we normalize
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
        validateIndices = np.random.choice(totalrealizations, np.floor(0.2 * totalrealizations), replace=False)
        trainIndices = all(np.array(range(totalrealizations)) != a for a in validateIndices)

        # Train the models
        results_lat = self.m_model_lat.fit(
            lat_X[trainIndices, :], lat_y[trainIndices, :],
            epochs=1000,
            batch_size=self.m_acquisition_length,
            validation_data=(lat_X[validateIndices, :], lat_y[validateIndices, :])
        )

        results_lon = self.m_model_lon.fit(
            lon_X[trainIndices, :], lon_y[trainIndices, :],
            epochs=1000,
            batch_size=self.m_acquisition_length,
            validation_data=(lon_X[validateIndices, :], lon_y[validateIndices, :])
        )

    def dump_nn_summary(self):
        # Dumps summary to debugger output and to log file saved in logs
        self.m_model_lat.summary()
        self.m_model_lon.summary()

    def nn_inference(self, path_latlon_noisy):
        # Perform inference on given path
        self.m_model_lat.predict(path_latlon_noisy[0])
        self.m_model_lon.predict(path_latlon_noisy[1])

    def save_models(self, modelname_lat, modelname_lon):
        # Save both models with unique name in directory NeuralNetworks/Models
        self.m_model_lat.save(modelname_lat)
        self.m_model_lon.save(modelname_lon)

    def load_models(self, modelname_lat, modelname_lon):
        # Loads both models from unique names from directory NeuralNetworks/Models
        self.m_model_lat = keras.models.load_model(modelname_lat)
        self.m_model_lon = keras.models.load_model(modelname_lon)

    def normalize_path_training(self, paths_latlon_org, paths_latlon_noisy):
        # Normalizing both downsampled and original data
        mean = np.repeat(np.expand_dims(np.mean(paths_latlon_org, axis=1), axis=1), self.number_of_samples, axis=1)
        var = np.repeat(np.expand_dims(np.var(paths_latlon_org, axis=1), axis=1), self.number_of_samples, axis=1)
        paths_latlon_org_norm = (paths_latlon_org - mean) / np.sqrt(var)

        mean = np.repeat(np.expand_dims(np.mean(paths_latlon_noisy, axis=1), axis=1), self.m_acquisition_length, axis=1)
        var = np.repeat(np.expand_dims(np.var(paths_latlon_noisy, axis=1), axis=1), self.m_acquisition_length, axis=1)
        paths_latlon_noisy_norm = (paths_latlon_noisy - mean) / np.sqrt(var)

        return paths_latlon_org_norm, paths_latlon_noisy_norm
