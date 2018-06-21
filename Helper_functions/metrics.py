# -*- coding: utf-8 -*-

## Import useful python modules
import numpy as np

## Main functions
def root_mean_square_deviation(true_array, test_array) : 
    return np.sqrt(np.mean((true_array[0]-test_array[0])**2+(true_array[1]-test_array[1])**2))
