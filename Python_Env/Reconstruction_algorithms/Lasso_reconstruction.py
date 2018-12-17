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

# Logging
import logging

logger = logging.getLogger('BckTrk')


class cLasso:
    # Constructor
    def __init__(self, struct):
        self.reconstruct_from_dct = struct['bReconstruct_from_dct']
        self.m_model = Lasso(alpha=struct['lasso_learning_rate'], tol=0.0001, fit_intercept=False, normalize=True)
        self.bUse_gaussian_matrix = struct['bUse_gaussian_matrix']

    # Reconstruction function
    def reconstructor(self, path, samples):
        if self.reconstruct_from_dct:
            logger.debug("LASSO reconstruction with DCT transform")

            acquisition_length = len(path)
            D = ft.dct(np.eye(acquisition_length), norm='ortho')

            if self.bUse_gaussian_matrix:
                A = np.dot(samples, D)  # Samples here is implicitly a randomly generated gaussian matrix
                y = np.dot(samples, path)
            else:
                A = D[samples]
                y = path[samples]

            self.m_model.fit(A, y)

            reconstructed_path = ft.idct(self.m_model.coef_, norm='ortho')
            reconstructed_path = reconstructed_path + np.mean(path[samples] - reconstructed_path[samples])

            return reconstructed_path
