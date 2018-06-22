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
## Python library import
import numpy as np  
import scipy.fftpack as ft
from sklearn.linear_model import Lasso
import sys

## Logging
import logging
logger = logging.getLogger('BckTrk')

## The main processing function
def lasso_algo(params, path): 
    algo_model = cLasso(params)
    return algo_model.reconstructor(path)

class cLasso:
    ## Constructor
    def __init__(self,struct):
        self.m_acquisition_length = struct['acquisition_length']
        self.m_model = Lasso(alpha= struct['lasso_learning_rate'])
        if struct["sampling_ratio"] > 1 :
            logger.error("Sampling_ratio larger than 1")
            sys.exit("Sampling_ratio larger than 1")
        self.number_of_samples = int(struct["sampling_ratio"]*struct["gps_freq_Hz"]*struct["acquisition_time_sec"])
        self.reconstruct_from_dct = struct['reconstruct_from_dct']
        
    
    ## Reconstruction function
    def reconstructor(self,path):
        if self.reconstruct_from_dct : 
            k = self.number_of_samples
            samples = np.sort(np.random.choice(self.m_acquisition_length,k,replace=False))
            D = ft.dct(np.eye(self.m_acquisition_length))
            A = D[samples]
            self.m_model.fit(A,path[samples])
            reconstructed_path = ft.idct(self.m_model.coef_)
            reconstructed_path = reconstructed_path + np.mean(path[samples]-reconstructed_path[samples])
            return reconstructed_path
            