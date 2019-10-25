# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Adaptive sampling algorithm
 Project     : Simulation environment for BckTrk app
 File        : Adaptive_sampling.py
 -----------------------------------------------------------------------------

   Description :


   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   10-Dec-2018  1.0      Taimir      File created
 =============================================================================

"""
# Python library import
import numpy as np
from Helper_functions.framework_error import CErrorTypes

# Logging
import logging

logger = logging.getLogger('BckTrk')


def checkParameter(parameter):
    if (parameter < 0) or (parameter > 1):
        logger.debug("Parameter value not inbound")
        errdict = {"file": __file__, "message": "Parameter value not inbound", "errorType": CErrorTypes.value}
        raise ValueError(errdict)


# Noise special bounds
def checkParameterNoise(epsilon_r):
    logger.debug("Bounds check on noise vector")
    for item in range(len(epsilon_r)):
        if epsilon_r[item] > 1:
            epsilon_r[item] = 1
        elif epsilon_r[item] < 0:
            logger.debug("Noise value at index <%d> < 0 ", item)
            errdict = {"file": __file__, "message": "Noise value < 0", "errorType": CErrorTypes.value}
            raise ValueError(errdict)
    return epsilon_r


class cAdaptiveSampling:
    # Constructor
    def __init__(self, struct, bUseAdaptive, noise_dist):
        self.m_acquisition_length = len(noise_dist)  # because we do not have access to higher layer of struct
        if "bUse_gaussian_matrix" in struct:
            self.bUse_gaussian_matrix = struct['bUse_gaussian_matrix']
        else:
            self.bUse_gaussian_matrix = False

        if struct['sampling_ratio'] > 1:
            logger.debug("Sampling_ratio larger than 1")
            errdict = {"file": __file__, "message": "Sampling_ratio larger than 1", "errorType": CErrorTypes.value}
            raise ValueError(errdict)

        self.sampling_ratio = struct['sampling_ratio']
        self.bAdaptiveSampling = bUseAdaptive  # no then random sampling
        if bUseAdaptive:
            self.threshold_high = struct['threshold_high']
            self.threshold_low = struct['threshold_low']
            self.forgetting_factor = struct['forgetting_factor']
            scaled_noise = noise_dist / struct['baseline']
            self.epsilon_r = checkParameterNoise(scaled_noise)
            self.cap_param = struct['cap_param']
            checkParameter(self.sampling_ratio)
            # checkParameter(self.threshold_high)
            checkParameter(self.threshold_low)
            checkParameter(self.forgetting_factor)
            checkParameter(self.cap_param)
            if self.threshold_high < self.threshold_low:
                raise ValueError
        else:
            self.threshold_high = 0
            self.threshold_low = 0
            self.forgetting_factor = 0
            self.epsilon_r = 0
            self.cap_param = 0

        self.gamma = np.zeros(self.m_acquisition_length)
        self.drawOutcome = np.zeros(self.m_acquisition_length)
        self.probability = np.zeros(self.m_acquisition_length)

    # Adaptive sampling
    def adaptiveSample(self):
        if self.bAdaptiveSampling:
            logger.debug("Adaptive sampling algorithm")

            firstIteration = True
            n = 0
            epsilon_s = 0

            while n < self.m_acquisition_length:
                if firstIteration:
                    firstIteration = False
                else:
                    if self.drawOutcome[n - 1] == 1:
                        epsilon_s = self.epsilon_r[n - 1]
                        self.gamma[n] = self.cap_param
                    else:
                        self.gamma[n] = self.forgetting_factor * self.gamma[n - 1]

                if epsilon_s >= self.threshold_high:
                    self.probability[n] = (1 - epsilon_s * self.gamma[n]) * self.sampling_ratio
                elif epsilon_s >= self.threshold_low:
                    self.probability[n] = self.sampling_ratio
                else:
                    self.probability[n] = self.sampling_ratio + (1 - self.sampling_ratio) * (1 - epsilon_s) * \
                                          self.gamma[n]

                # Update step
                self.drawOutcome[n] = np.random.binomial(1, self.probability[n], 1)
                n = n + 1

            final_sampling_ratio = np.count_nonzero(self.drawOutcome) / self.m_acquisition_length
            samples = np.sort(np.nonzero(self.drawOutcome))
            samples = samples[0]  # Because nonzero returns a tuple
        else:
            logger.debug("Random sampling to speed up process")

            final_sampling_ratio = self.sampling_ratio
            number_of_samples = int(self.sampling_ratio * self.m_acquisition_length)
            if number_of_samples <= 0:
                logger.debug("Number of samples cannot be 0 or negative")
                errdict = {"file": __file__, "message": "Invalid number of samples", "errorType": CErrorTypes.value}
                raise ValueError(errdict)
            if self.bUse_gaussian_matrix:
                gaussian_matrix = np.random.normal(0, 1, [number_of_samples,
                                                          self.m_acquisition_length]) / self.m_acquisition_length
                samples = gaussian_matrix
            else:
                #samples = np.sort(np.random.choice(self.m_acquisition_length, number_of_samples, replace=False))
                samples = np.arange(0, number_of_samples, 1)

        return samples, final_sampling_ratio
