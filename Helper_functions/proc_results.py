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
logger = logging.getLogger('BckTrk')
    
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
        with open (self.m_filename, 'wb') as txt_file:
            pickle.dump(self.m_localStruct, txt_file)
            
    ## Read from txt file pickle format into dictionary        
    def get_pickle_file(self):
        with open (self.m_filename, 'rb' ) as txt_file_read :
            return pickle.load (txt_file_read)
    
    ## Real path in 2D plotting
    def plot_real_path_2d(self) : 
        #Set of params
        paths_wm_org = self.m_localStruct['RESULTS']['paths_wm_org']
        paths_latlon_org = self.m_localStruct['RESULTS']['paths_latlon_org']
        
        acquisition_length = self.m_localStruct['gps_freq_Hz']*self.m_localStruct['acquisition_time_sec']
        x_axis = range(0, acquisition_length)
        number_realization = self.m_localStruct ['realization']

        ####################
        logger.info ('Plotting original path in webmercator coordinates')
        logger.warning ('Plotting only first realization for visibility')
        for i in range (number_realization):
            plt.plot(paths_wm_org[0,:,0],paths_wm_org[1,:,0],'b-*')
            
        plt.grid()
        plt.title('Original cartesian Path')
        plt.xlabel('x-coordinates')
        plt.ylabel('y-coordinates')
        plt.show() 
        
        ####################
        logger.info ('Plotting original path in webmercator coordinates')
        logger.warning ('Plotting only first realization for visibility')
        #Plotting x coordinates
        for i in range (number_realization):
            plt.plot(x_axis,paths_wm_org[0,:,0],'b-*')
            
        plt.grid()
        plt.title('Original webmercator Path -X')
        plt.xlabel('Number of steps')
        plt.ylabel('x-coordinates')
        plt.show()  
        
        #Plotting y coordinates
        for i in range (number_realization):
            plt.plot(x_axis, paths_wm_org[1,:,0],'r-*')
            
        plt.grid()
        plt.title('Original webmercator Path -Y')
        plt.xlabel('Number of steps')
        plt.ylabel('y-coordinates')
        plt.show() 
        
        ####################
        logger.info ('Plotting original longitude and lattitude')
        logger.warning ('Plotting only first realization for visibility')
          
        #Plotting Latitude
        for i in range (number_realization):
            plt.plot(x_axis, paths_latlon_org[0,:,0],'r-*')
            
        plt.grid()
        plt.title('Original lattitude')
        plt.xlabel('Number of steps')
        plt.ylabel('Lattitude')
        plt.show() 
            
        #Plotting Longitude
        for i in range (number_realization):
            plt.plot(x_axis,paths_latlon_org[1,:,0],'b-*')
            
        plt.grid()
        plt.title('Original longitude')
        plt.xlabel('Number of steps')
        plt.ylabel('Longitude')
        plt.show()