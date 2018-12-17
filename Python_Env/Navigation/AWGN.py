# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Generate and add noise to incoming data
 Project     : Simulation environment for BckTrk app
 File        : AWGN.py
 -----------------------------------------------------------------------------

   Description :


   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   26-Mai-2018  1.0      Rami      File created
 =============================================================================
 
"""

import numpy as np
import Navigation.Coordinates as cord
from Helper_functions.framework_error import CFrameworkError
from Helper_functions.framework_error import CErrorTypes

# Bring the logger
import logging

logger = logging.getLogger('BckTrk')


# the main processing function
def noise_generator(params, positions_wm, noise_level):
    data_obj = cAWGN(params, noise_level)
    data_obj.generate_noisy_signal_dist()
    data_obj.add_noise_signal(positions_wm)
    return data_obj.m_noisy_positions_wm, data_obj.m_noisy_positions_latlon, data_obj.noise_dist


class cAWGN:
    # Constructor
    def __init__(self, struct, noise_level):
        logger.debug("Initializing cAWGN")
        self.m_acquisition_length = struct['acquisition_length']
        self.m_noise_std = struct['noise_std_meter']
        self.m_noisy_positions_latlon = np.zeros((2, self.m_acquisition_length))
        self.m_noisy_positions_wm = np.zeros((2, self.m_acquisition_length))
        self.m_noise_level = noise_level
        self.noise = np.transpose(
            np.random.multivariate_normal([0, 0], [[1 ** 2, 0], [0, 1 ** 2]], self.m_acquisition_length))
        self.noise_dist = np.zeros(self.m_acquisition_length)

    # noise generation based on the distance
    def generate_noisy_signal_dist(self):
        logger.debug("Generating noisy signal on distance")
        isFirstChunk = True
        number_of_chunks = len(self.m_noise_std)

        chunkIndex = np.arange(number_of_chunks)
        np.random.shuffle(self.m_noise_std)
        try:
            chunk_lens = self.generate_rand_chunk(number_of_chunks)
        except ValueError as valerr:
            logger.debug('All chunk lengths are zero')
            errdict = {"file": __file__, "message": valerr.args[0], "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict) from valerr

        for chunk in chunkIndex:
            chunk_len = int(chunk_lens[chunk])
            if isFirstChunk:
                debut = 0
                fin = chunk_len
                isFirstChunk = False
            else:
                debut = debut + int(chunk_lens[chunk - 1])
                fin = chunk_len + debut

            noise_dist_chunk = np.random.normal(loc=self.m_noise_level, scale=self.m_noise_std[chunk], size=chunk_len)
            # Concatenate back the chunks
            self.noise_dist[debut:fin] = np.abs(noise_dist_chunk)
        try:
            self.noise = self.noise * self.noise_dist
        except ValueError as valerr:
            logger.debug('Invalid noise length <%s>', valerr.args[0])
            errdict = {"file": __file__, "message": valerr.args[0], "errorType": CErrorTypes.value}
            raise CFrameworkError(errdict) from valerr

    # add noise to signal and convert back to latlon
    def add_noise_signal(self, positions_wm):
        logger.debug("Applying noise to signal and converting back to latlon")
        self.m_noisy_positions_wm = positions_wm + self.noise
        self.m_noisy_positions_latlon = cord.generate_latlon_array(self.m_noisy_positions_wm)

    # generate random chunks lengths
    def generate_rand_chunk(self, number_of_chunks):
        logger.debug("Generating random chunk lengths for different std")
        chunk_lens = np.zeros(number_of_chunks)
        left_length = int(self.m_acquisition_length)
        for item in range(number_of_chunks):
            if not left_length == 0:
                if item == number_of_chunks - 1:
                    chunk_lens[item] = left_length
                else:
                    chunk_lens[item] = np.random.randint(0, left_length)
                    left_length = left_length - chunk_lens[item]
        if not np.any(chunk_lens):
            raise ValueError
        else:
            return chunk_lens
