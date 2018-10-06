# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Lasso reconstruction
 Project     : Simulation environment for BckTrk app
 File        : Lasso_reconstruction.py
 -----------------------------------------------------------------------------

   Description :

 
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   21-Jun-2018  1.0      Ramy      File created
 =============================================================================
 
"""
# Python library import
import numpy as np
import scipy.fftpack as ft
from sklearn.linear_model import Lasso
from Helper_functions.framework_error import CErrorTypes

## Logging
import logging

logger = logging.getLogger('BckTrk')


# The main processing function
def lasso_algo(params, path):
    algo_model = cLasso(params)
    return algo_model.reconstructor(path)


class cLasso:
    # Constructor
    def __init__(self, struct):

        if struct['RCT_ALG_LASSO']["sampling_ratio"] > 1:
            logger.debug("Sampling_ratio larger than 1")
            errdict = {"file": __file__, "message": "Sampling_ratio larger than 1", "errorType": CErrorTypes.value}
            raise ValueError(errdict)

        self.m_acquisition_length = struct['acquisition_length']
        self.m_model = Lasso(alpha=struct['RCT_ALG_LASSO']['lasso_learning_rate'], tol=0.0001, fit_intercept=False,
                             normalize=True)

        self.number_of_samples = int(struct['RCT_ALG_LASSO']["sampling_ratio"] * struct["acquisition_length"])
        self.reconstruct_from_dct = struct['RCT_ALG_LASSO']['bReconstruct_from_dct']

        if self.number_of_samples <= 0:
            logger.debug("Number of samples cannot be 0 or negative")
            errdict = {"file": __file__, "message" : "Invalid number of samples", "errorType": CErrorTypes.value }
            raise ValueError(errdict)

    # Reconstruction function
    def reconstructor(self, path):
        if self.reconstruct_from_dct:
            logger.debug("LASSO reconstruction with DCT transform")

            k = self.number_of_samples
            samples = np.sort(np.random.choice(self.m_acquisition_length, k, replace=False))
            D = ft.dct(np.eye(self.m_acquisition_length), norm='ortho')
            A = D[samples]

            self.m_model.fit(A, path[samples])

            reconstructed_path = ft.idct(self.m_model.coef_, norm='ortho')
            reconstructed_path = reconstructed_path + np.mean(path[samples] - reconstructed_path[samples])

            return reconstructed_path
