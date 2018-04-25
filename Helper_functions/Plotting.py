# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:51:23 2018

@author: Rami Plotting script 
"""

import matplotlib.pyplot as plt


def plot_real_path_2d(path2d, params) : #the params should be an instance of the parameters class defined in config.py to access specific variables you need to look in the dictionary of the "variables" attribute
    plt.plot(path2d[0],path2d[1],'r.-',lw=3)
    plt.xlim(params.variables['xlimits'][0], params.variables['xlimits'][1])
    plt.ylim(params.variables['ylimits'][0], params.variables['ylimits'][1])
    plt.grid()
    plt.title('Real Path')
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.show()

    