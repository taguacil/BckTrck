# %-*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:19:47 2018

@author: Rami This will be the main script 
"""

import numpy as np
import Helper_functions.Plotting as plotting #this is the plotting helper module
import sys
import Config
import json

local_parameters = Config.parameters()

    
if len(sys.argv) < 2:
        print('We are in the if')
else:
        print ('We are in the else')
        parameters_filename = sys.argv[1] #first argument after the main script name should the parameters as a text file
        new_dictionary = json.load(open(parameters_filename)) #JSON loads a dictionary written in a text file as a python dictionary
        for key in new_dictionary :
            local_parameters.variables[key] = new_dictionary[key]

def main() :
            
    x = np.random.randint(0,100,size=(2,20))
    plotting.plot_real_path_2d(x, local_parameters)
    
if __name__ == "__main__" :
    main()
    
