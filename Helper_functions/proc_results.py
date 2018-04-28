# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Plotting script
 Project     : Simulation environment for BckTrk app
 File        : proc_results.py
 -----------------------------------------------------------------------------

   Description :

   This file is responsible for processing all the result including 
   all plotting in the framework and save the config to a file
   It must be able to handle different plots with different requirements
   Save options and some flags for selective plotting
   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   25-Apr-2018  1.0      Rami      File created
 =============================================================================
 
"""
## Python library import
import matplotlib.pyplot as plt
import _pickle  as pickle

## Bring the logger
import logging
logger = logging.getLogger(__name__)
    
## the main processing function 
def process_data (params):
    data_obj = cProcessFile (params)
    data_obj.set_pickle_file()
    if (params ['bPlotRealPath']):
        data_obj.plot_real_path_2d()
        
class cProcessFile:
    ## Constructor
    def __init__(self,struct):
        self.m_localStruct = struct
        self.m_filename =  struct['workingDir'] + '\\Results\\' + 'BckTrk_Res_' + struct['currentTime'].strftime("%Y-%m-%d")+'.txt'  
        
    ## Write dictionary into pickle format to txt file    
    def set_pickle_file(self) :
        with open (self.filename, 'wb') as txt_file:
            pickle.dump(self.m_localStruct, txt_file)
            
    ## Read from txt file pickle format into dictionary        
    def get_pickle_file(self):
        with open (self.m_filename, 'rb' ) as txt_file_read :
            return pickle.load (txt_file_read)
    
    ## Real path in 2D plotting
    def plot_real_path_2d(self) : 
        #Set of params
        path2d = self.m_localStruct['RESULTS']['data_2d']
        xlim = self.m_localStruct['xlimits']
        ylim = self.m_localStruct ['ylimits']

        #Plotting
        plt.plot(path2d[0],path2d[1],'r.-',lw=3)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.grid()
        plt.title('Real Path')
        plt.xlabel('x-coordinates')
        plt.ylabel('y-coordinates')
        plt.show()    
            
