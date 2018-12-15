# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : BFGS reconstruction
 Project     : Simulation environment for BckTrk app
 File        : BFGS.py
 -----------------------------------------------------------------------------

   Description :

 
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   21-Jul-2018  1.0      Taimir      File created
 =============================================================================
 
"""
# Python library import
import numpy as np
import scipy.fftpack as ft
import scipy.optimize as opt

# Logging
import logging

logger = logging.getLogger('BckTrk')


class cBFGS:
    # Constructor
    def __init__(self, struct):

        self.bUse_gaussian_matrix = struct['bUse_gaussian_matrix']
        self.m_lambda_param = struct['lambda']
        self.m_u = struct['u']
        self.m_reconstruct_from_dct = struct['bReconstruct_from_dct']
        self.m_maxiter = struct['maxiter']

        self.m_A = 0
        self.m_y = 0
        self.bUseSciGradient = True  # HardCoded

    # argument definition
    def bfgs_run(self, path, samples):
        logger.debug("BFGS starting")

        acquisition_length = len(path)

        D = ft.dct(np.eye(acquisition_length), norm='ortho')
        # Initial values
        x0 = np.random.normal(0, 1, acquisition_length)

        if self.bUse_gaussian_matrix:
            self.m_A = np.dot(samples, D)
            self.m_y = np.dot(samples, path)
        else:
            self.m_A = D[samples]
            self.m_y = path[samples]

        if self.bUseSciGradient:
            reconst_path = self.scipy_func(x0, D)
        else:
            reconst_path = self.reconstructor(x0, D)

        return reconst_path

    # scipi
    def scipy_func(self, x0, D):
        xopt = opt.fmin_bfgs(self.cost_fun, x0, maxiter=self.m_maxiter, norm=1)
        return np.dot(D, xopt)

    # Objective function
    def cost_fun(self, x):
        A = self.m_A
        y = self.m_y

        linear_OP = np.dot(A, x) - y
        norm_sq = (np.linalg.norm(linear_OP)) ** 2
        regul = self.m_lambda_param * (np.linalg.norm(x, 1))

        return norm_sq + regul

    # shitty Gradient function
    """
    def gradient(self,x):
        A=self.m_A
        y=self.m_y
        u=self.m_u
        
        A_her=A.conj().T
        linear_OP=np.dot(A,x)-y
        delta=np.dot(np.conj(x),x)+u
        regul=self.m_lambda_param*(x/np.sqrt(delta))
        
        return 2*np.dot(A_her,linear_OP)+regul
    """

    def gradient(self, xk, epsilon=1e-8):
        f0 = self.cost_fun(*((xk,)))
        grad = np.zeros((len(xk),), float)
        ei = np.zeros((len(xk),), float)
        for k in range(len(xk)):
            ei[k] = 1.0
            d = epsilon * ei
            grad[k] = (self.cost_fun(*((xk + d,))) - f0) / d[k]
            ei[k] = 0.0

        return grad

    # Reconstruction function
    def reconstructor(self, x0, D):
        if self.m_reconstruct_from_dct:
            logger.debug("BFGS reconstruction with DCT transform")

        # Initialization
        maxiter = self.m_maxiter
        tolgrad = 1e-10
        alpha = 0.55
        beta = 0.4
        k = 0
        Armijo_search = 20
        n = x0.shape[0]
        B = np.eye(n)
        g = self.gradient(x0)

        # While loop beginning
        while (np.linalg.norm(g) > tolgrad) & (k < maxiter):
            logger.info('Iteration <%d>', k)
            print('Iteration <%d>' % (k))

            d = -np.dot(np.linalg.pinv(B), g)
            m = 0
            m_k = 0

            # Armijo rule to line search
            while m < Armijo_search:
                left_exp = self.cost_fun(x0 + (alpha ** m) * d)
                right_exp = self.cost_fun(x0) + beta * (alpha ** m) * np.dot(np.conj(g), d)
                if (left_exp < right_exp):
                    m_k = m
                    break
                m = m + 1
            # BFGS adjustment
            x = x0 + (alpha ** m_k) * d
            s = x - x0
            y_k = self.gradient(x) - g
            if np.dot(y_k, s) > 0:
                elem1 = np.dot(B, np.outer(s, np.dot(np.conj(s), B))) / np.dot(np.conj(s), np.dot(B, s))
                elem2 = (np.outer(y_k, np.conj(y_k))) / (np.dot(np.conj(y_k), s))
                B = B - elem1 + elem2
            else:
                B = B

            x0 = x
            g = self.gradient(x0)
            k = k + 1

        # output
        reconstructed_path = np.dot(D, x0)
        # check cost value
        print(self.cost_fun(x0))

        return reconstructed_path
