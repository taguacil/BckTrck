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
## Python library import
import numpy as np  
import scipy.fftpack as ft
import scipy.optimize as opt

import matplotlib.pyplot as plt

## Logging
import logging
logger = logging.getLogger('BckTrk')

## The main processing function
def bfgs_algo(): 
    #Path generation
    xlim=256
    ratio=0.1
    x = np.arange(0,xlim,1)
    path=(1/np.sqrt(2))*np.cos(x)+(1/np.sqrt(2))*1j*np.cos(x)
    path=np.cos(x)+1j*np.cos(x)
    path=path.real
    
    #Class initialization
    algo_model = cBFGS(path,xlim,ratio)
    
    #Objective function value for ideal case 
    D_inv = ft.idct(np.eye(256),norm='ortho')
    print(algo_model.cost_fun(np.dot(D_inv,path)))
    
    #Reconstruct with scipyBFGS
    reconst=algo_model.scipy_func()
     
    #Reconstruct with homemade
    #reconst= algo_model.reconstructor()
   
    #Display SNR
    SNR = 20*np.log10(np.linalg.norm(path,1)/np.linalg.norm(path-reconst,1))
    print('%3f dB'%(SNR))
    
    #Plot
    plt.plot(x,path.real,'b-*')
    plt.plot(x,reconst.real,'r-*')
    plt.show()
    
    plt.plot(x,path.imag,'b-*')
    plt.plot(x,reconst.imag,'r-*')
    plt.show()
    
    
class cBFGS:
    ## Constructor
    def __init__(self,path,acq,ratio):
         
        self.m_acquisition_length       = acq
        self.lambda_param               = 0.01
        self.u                          = 10e-6
        self.number_of_samples          = int(acq*ratio)
        self.reconstruct_from_dct       = 1
        
        #Gaussian sampling matrix
        self.gaussian_matrix            = np.random.normal(0,1,[self.number_of_samples,self.m_acquisition_length])/self.m_acquisition_length
        self.D                          = ft.dct(np.eye(self.m_acquisition_length),norm='ortho')
        self.y                          = np.dot(self.gaussian_matrix,path)
        self.A                          = np.dot(self.gaussian_matrix,self.D)
        
        #Drop randomly samples
        samples                         = np.sort(np.random.choice(self.m_acquisition_length,self.number_of_samples,replace=False))
        self.y                          = path[samples]
        self.A                          = self.D[samples]
        
        #Initial values
        self.x0                         = np.random.normal(0,1,self.m_acquisition_length)
    
    
    ## scipi
    def scipy_func(self):
        xopt = opt.fmin_bfgs(self.cost_fun,self.x0,maxiter=500,norm=1)
        return np.dot(self.D,xopt)
    
    ## Objective function
    def cost_fun(self,x):
        A=self.A
        y=self.y
        
        linear_OP=np.dot(A,x)-y
        norm_sq=(np.linalg.norm(linear_OP))**2
        regul=self.lambda_param*(np.linalg.norm(x,1))
        
        return norm_sq+regul
    
    ## shitty Gradient function
    def gradient(self,x):
        A=self.A
        y=self.y
        u=self.u
        
        A_her=A.conj().T
        linear_OP=np.dot(A,x)-y
        delta=np.dot(np.conj(x),x)+u
        regul=self.lambda_param*(x/np.sqrt(delta))
        
        return 2*np.dot(A_her,linear_OP)+regul
    
    ## Reconstruction function
    def reconstructor(self):
        if self.reconstruct_from_dct : 
            logger.debug("BFGS reconstruction with DCT transform")
        
        ## Initialization
        maxiter=500 
        tolgrad=1e-10 
        alpha=0.55
        beta=0.4 
        k=0
        Armijo_search = 20
        x0=self.x0
        n=x0.shape[0]
        B=np.eye(n)
        g=self.gradient(x0)
    
        ## While loop beginning
        while (np.linalg.norm(g)>tolgrad) & (k<maxiter):
            logger.info('Iteration <%d>',k)
            print('Iteration <%d>'%(k))
            
            d = -np.dot(np.linalg.pinv(B),g)
            m = 0
            m_k = 0
            
            ## Armijo rule to line search
            while m<Armijo_search :
                left_exp=self.cost_fun(x0 + (alpha**m)*d)
                right_exp=self.cost_fun(x0) + beta*(alpha**m)*np.dot(np.conj(g),d)
                if (left_exp < right_exp) :
                    m_k = m 
                    break
                m = m+1
            ## BFGS adjustment    
            x = x0 + (alpha**m_k)*d 
            s = x-x0
            y_k = self.gradient(x)-g
            if np.dot(y_k,s) > 0 :
                elem1= np.dot(B, np.outer(s, np.dot(np.conj(s),B)))/np.dot(np.conj(s),np.dot(B,s))
                elem2= (np.outer(y_k, np.conj(y_k)))/(np.dot(np.conj(y_k),s)) 
                B = B -  elem1 + elem2 
            else :
                B = B
                
            x0 = x
            g = self.gradient(x0)
            k = k+1
       
        ## output 
        reconstructed_path = np.dot(self.D,x0)
        #check cost value
        print(self.cost_fun(x0))
        
        return reconstructed_path
    
bfgs_algo() 