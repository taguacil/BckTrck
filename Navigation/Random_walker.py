# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 06:09:24 2018

@author: Rami
"""

import numpy as np 

def random_2d_path_generator(params):
    """ generates a random 2d walk over a grid, can only move in cardinal directions, the stepsize in normalized to 1 to avoid numerical precision problems, returns a 2d array with the x-positions on one axis and the y-positions on the other """
    
    number_of_steps = params['number_of_steps']
    random_seed = params['random_seed']
    use_random_seed = params['use_random_seed'] # Controls whether we want to use our random seed or let numpy initialize the randomstate "randomly"
    
    if use_random_seed :
        np.random.seed(random_seed)
    
    positions = np.zeros((2,number_of_steps))
    
    for i in range(1,number_of_steps) :
        chance = np.random.randint(0,4)
        if chance == 0 :
            positions[:,i] = positions[:,i-1] + np.array([1,0])
        if chance == 1 : 
            positions[:,i] = positions[:,i-1] + np.array([0,1])
        if chance == 2 :
            positions[:,i] = positions[:,i-1] + np.array([-1,0])
        if chance == 3 :
            positions[:,i] = positions[:,i-1] + np.array([0,-1])
        
    return positions 

