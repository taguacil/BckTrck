# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.fftpack as ft
from sklearn.linear_model import Lasso

def path(init,stepsize,n):
    path = np.zeros((2,n))
    path[:,0] = init
    for i in range(n-1):
        theta = np.random.uniform(0,np.pi)
        path[:,i+1] = path[:,i] + stepsize*np.array([np.cos(theta),np.sin(theta)])
    return path

## DCT 
#Average root mean square deviation for n=100
    
average_rms = 0
average_sampling = 0

for i in range(500):
    n = 1000
    generated_path = path(np.array([0,0]),1,n)
    x = generated_path[0]
    y = generated_path[1]
    transformed_x = ft.dct(x)
    transformed_y = ft.dct(y)
    k= 100
    samples = np.random.randint(0,n,(k,))
    samples = np.unique(np.sort(samples))
    average_sampling += len(samples)/float(n)
    D = ft.dct(np.eye(n))
    A = D[samples]
    lasso = Lasso(alpha=0.01)
    lasso.fit(A,x[samples])
    xprime = ft.idct(lasso.coef_)
    lasso = Lasso(alpha=0.01)
    lasso.fit(A,y[samples])
    yprime = ft.idct(lasso.coef_)
    xprime = xprime + np.mean(x[samples] - xprime[samples])
    yprime = yprime + np.mean(y[samples] - yprime[samples])
    average_rms += np.sqrt(np.mean((y-yprime)**2+(x-xprime)**2))

average_rms /= 500
average_sampling /= 500

print("Average root mean square deviation is %.3f m"%(average_rms))   
print("Average sampling %.2f %% of path"%(average_sampling*100))