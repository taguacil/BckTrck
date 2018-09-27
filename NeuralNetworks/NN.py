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

# User-defined library import
from Helper_functions.framework_error import CFrameworkError
from Helper_functions.framework_error import CErrorTypes

# Logging
import logging

logger = logging.getLogger('BckTrk')

class CNeuralNetwork:
    # Constructor
    def __init__(self, struct):

        if struct['RCT_ALG_NN']["sampling_ratio"] > 1:
            logger.debug("Sampling_ratio larger than 1")
            errdict = {"file": __file__, "message": "Sampling_ratio larger than 1", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.m_acquisition_length = struct['acquisition_length']
        if struct['bTrainNetwork']:
            self.m_model = keras.Sequential()

        self.number_of_samples = int(struct['RCT_ALG_LASSO']["sampling_ratio"] * struct["acquisition_length"])
        self.realizations = struct['realization']

        if self.number_of_samples <= 0:
            logger.debug("Number of samples cannot be 0 or negative")
            errdict = {"file": __file__, "message": "Invalid number of samples", "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict)

        self.alpha = struct['RCT_ALG_NN']["alpha"]

    def activation_fun(self, x):
        return keras.activations.relu(x, alpha=self.alpha)

    def design_nn(self):
        self.model.add(
            keras.layers.Dense(self.m_acquisition_length, activation=self.activation_fun,
                               input_shape=(self.number_of_samples,)))
        self.model.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.model.add(keras.layers.Dropout(0.1))
        self.model.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.model.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.model.add(keras.layers.Dropout(0.1))
        self.model.add(keras.layers.Dense(50, activation=self.activation_fun))
        self.model.add(keras.layers.Dense(250, activation=self.activation_fun))
        self.model.add(keras.layers.Dense(self.m_acquisition_length, activation="linear"))

        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"]
        )

    def train_nn(self):
        results = self.model.fit(
            train_X, train_y,
            epochs=1000,
            batch_size=500,
            validation_data=(test_X, test_y)
        )

    def dump_nn_summary(self):
        self.model.summary()

    def nn_inference(self):
        self.model.predict(test_X[0].reshape((1, 100)))

    def save_model(self, modelname):
        self.model.save(modelname)

    def load_model(self, modelname):
        self.model = keras.models.load_model(modelname)
