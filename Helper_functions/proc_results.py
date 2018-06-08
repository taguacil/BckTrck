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
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import _pickle  as pickle

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')
    
## the main processing function 
def process_data (params):
    data_obj = cProcessFile (params)
    data_obj.set_pickle_file()
    data_obj.plot_path_org_2d()
    data_obj.plot_path_noisy_2d()
    data_obj.plot_MSE()
    
class cProcessFile:
    ## Constructor
    def __init__(self,struct):
        self.m_localStruct = struct
        self.m_filename =  struct['workingDir'] + '\\Results\\' + 'BckTrk_Res_' + struct['currentTime'].strftime("%Y-%m-%d")+'.txt'  
        self.m_paths_wm_org = self.m_localStruct['RESULTS']['paths_wm_org']
        self.m_paths_latlon_org = self.m_localStruct['RESULTS']['paths_latlon_org']
        self.m_paths_wm_noisy = self.m_localStruct['RESULTS']['paths_wm_noisy']
        self.m_paths_latlon_noisy = self.m_localStruct['RESULTS']['paths_latlon_noisy']
        self.m_acquisition_length = self.m_localStruct['gps_freq_Hz']*self.m_localStruct['acquisition_time_sec']
        self.m_number_realization = self.m_localStruct ['realization']
        
    ## Write dictionary into pickle format to txt file    
    def set_pickle_file(self) :
        with open (self.m_filename, 'wb') as txt_file:
            pickle.dump(self.m_localStruct, txt_file)
            
    ## Read from txt file pickle format into dictionary        
    def get_pickle_file(self):
        with open (self.m_filename, 'rb' ) as txt_file_read :
            return pickle.load (txt_file_read)
    
    ## Original path in 2D plotting
    def plot_path_org_2d(self) : 
        #Set of params
        x_axis = range(0, self.m_acquisition_length)

        ####################
        if self.m_localStruct['bPlotPath_WM_org']:
            logger.info ('Plotting original path in webmercator coordinates')
            if self.m_localStruct['bPlotAllrealizations']:
                for k in range (self.m_number_realization):
                    plt.plot(self.m_paths_wm_org[0,:,k],self.m_paths_wm_org[1,:,k],'b-*')
            else:
                plt.plot(self.m_paths_wm_org[0,:,0],self.m_paths_wm_org[1,:,0],'b-*')
                logger.warning ('Plotting only first realization for visibility')
        
            plt.grid()
            plt.title('Original cartesian Path')
            plt.xlabel('x-coordinates')
            plt.ylabel('y-coordinates')
            plt.show() 
        
        ####################
        if self.m_localStruct['bPlotWM_time_org']:
            logger.info ('Plotting original path in webmercator coordinates')
            if self.m_localStruct['bPlotAllrealizations']:
                for k in range (self.m_number_realization):
                    plt.plot(x_axis,self.m_paths_wm_org[0,:,k],'b-*')
                     
            else:
                logger.warning ('Plotting only first realization for visibility') 
                plt.plot(x_axis,self.m_paths_wm_org[0,:,0],'b-*')    
        
            #Plotting x coordinates
            plt.grid()
            plt.title('Original webmercator Path -X')
            plt.xlabel('Number of steps')
            plt.ylabel('x-coordinates')
            plt.show()  
        
            #Plotting y coordinates
            if self.m_localStruct['bPlotAllrealizations']:
                for k in range (self.m_number_realization):
                    plt.plot(x_axis,self.m_paths_wm_org[1,:,k],'r-*')
                     
            else:
                plt.plot(x_axis,self.m_paths_wm_org[1,:,0],'r-*')
            
            plt.grid()
            plt.title('Original webmercator Path -Y')
            plt.xlabel('Number of steps')
            plt.ylabel('y-coordinates')
            plt.show() 
        
        ####################
        if self.m_localStruct['bPlotLonLat_time_org']:
            logger.info ('Plotting original longitude and latitude')
            if self.m_localStruct['bPlotAllrealizations']:
                for k in range (self.m_number_realization):
                    plt.plot(x_axis,self.m_paths_latlon_org[0,:,k],'r-*')
                     
            else:
                logger.warning ('Plotting only first realization for visibility')
                plt.plot(x_axis,self.m_paths_latlon_org[0,:,0],'r-*')
    
            #Plotting Latitude            
            plt.grid()
            plt.title('Original lattitude')
            plt.xlabel('Number of steps')
            plt.ylabel('Latitude')
            plt.show() 
            
            #Plotting Longitude
            if self.m_localStruct['bPlotAllrealizations']:
                for k in range (self.m_number_realization):
                    plt.plot(x_axis,self.m_paths_latlon_org[1,:,k],'b-*')
                     
            else:
                logger.warning ('Plotting only first realization for visibility')
                plt.plot(x_axis,self.m_paths_latlon_org[1,:,0],'b-*')
            
            plt.grid()
            plt.title('Original longitude')
            plt.xlabel('Number of steps')
            plt.ylabel('Longitude')
            plt.show()
                
    ## Noisy path in 2D plotting
    def plot_path_noisy_2d(self) : 
        #Set of params
        x_axis = range(0, self.m_acquisition_length)
        noise_level = self.m_localStruct['noise_level']
        
        for noise in range(len(noise_level)):
            paths_wm_noisy      = self.m_paths_wm_noisy[:,:,:,noise]
            paths_wm_org        = self.m_paths_wm_org
            paths_latlon_noisy  = self.m_paths_latlon_noisy[:,:,:,noise]
            paths_latlon_org    = self.m_paths_latlon_org
            
            if self.m_localStruct['bPlotPath_WM_noisy']:
                logger.info ('Plotting noisy path in webmercator coordinates')
                if self.m_localStruct['bPlotAllrealizations']:
                   x = paths_wm_noisy[0,:,:]-paths_wm_org[0,:,:]
                   y = paths_wm_noisy[1,:,:]-paths_wm_org[1,:,:]
                   h = plt.hist2d(x.flatten(),y.flatten(),100,norm=LogNorm()) 
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    x = paths_wm_noisy[0,:,0]-paths_wm_org[0,:,0]
                    y = paths_wm_noisy[1,:,0]-paths_wm_org[1,:,0]
                    h = plt.hist2d(x,y,100,norm=LogNorm())
                   
                buf = "Cartesian noise distribution for noise level %d (meters)" % (noise_level[noise])    
                plt.colorbar(h[3])    
                plt.grid()
                plt.title(buf)
                plt.xlabel('x-coordinates')
                plt.ylabel('y-coordinates')
                plt.show() 
            
                    ####################
            if self.m_localStruct['bPlotPath_LonLat_noisy']:
                logger.info ('Plotting noisy path in lonlat coordinates')
                if self.m_localStruct['bPlotAllrealizations']:
                    x = paths_latlon_noisy[0,:,:]-paths_latlon_org[0,:,:]
                    y = paths_latlon_noisy[1,:,:]-paths_latlon_org[1,:,:]
                    h = plt.hist2d(x.flatten(),y.flatten(),100,norm=LogNorm()) 
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    x = paths_latlon_noisy[0,:,0]-paths_latlon_org[0,:,0]
                    y = paths_latlon_noisy[1,:,0]-paths_latlon_org[1,:,0]
                    h = plt.hist2d(x,y,100,norm=LogNorm())
                
                buf = "LatLon noise distribution for noise level %d (meters)" % (noise_level[noise])
                plt.colorbar(h[3])    
                plt.grid()
                plt.title(buf)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.show() 
                
            ####################
            if self.m_localStruct['bPlotWM_time_noisy']:
                logger.info ('Plotting noisy path in webmercator coordinates')
                if self.m_localStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis,paths_wm_noisy[0,:,k],'b-*')
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    plt.plot(x_axis,paths_wm_noisy[0,:,0],'b-*')
                    
                #Plotting x coordinates noisy
                buf = "Noisy webmercator Path -X for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.xlabel('Number of steps')
                plt.ylabel('x-coordinates')
                plt.show()  
            
                if self.m_localStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis, paths_wm_noisy[1,:,k],'r-*')
                else:
                     plt.plot(x_axis, paths_wm_noisy[1,:,0],'r-*')
            
                #Plotting y coordinates
                buf = "Noisy webmercator Path -Y for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.xlabel('Number of steps')
                plt.ylabel('y-coordinates')
                plt.show() 
            
            ####################
            if self.m_localStruct['bPlotLonLat_time_noisy']:
                logger.info ('Plotting noisy longitude and latitude')
                if self.m_localStruct['bPlotAllrealizations']:
                    for i in range (self.m_number_realization):
                        plt.plot(x_axis, paths_latlon_noisy[0,:,k],'r-*')
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    plt.plot(x_axis, paths_latlon_noisy[0,:,0],'r-*')
                    
                #Plotting Latitude
                buf = "Noisy latitude for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.xlabel('Number of steps')
                plt.ylabel('Latitude')
                plt.show() 
                
                if self.m_localStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis,paths_latlon_noisy[1,:,k],'b-*')
                else:
                    plt.plot(x_axis,paths_latlon_noisy[1,:,0],'b-*')
            
                #Plotting Longitude
                buf = "Noisy longitude for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.xlabel('Number of steps')
                plt.ylabel('Longitude')
                plt.show()
            
    ## Plot MSE - mean error rate
    def plot_MSE(self) : 
        if self.m_localStruct['bPlotDER']: 
            #Set of params
            x_axis = self.m_localStruct['noise_level']
            paths_wm_org_ext=np.transpose(np.array([self.m_paths_wm_org,]*len(x_axis)),(1,2,3,0))
            paths_latlon_org_ext=np.transpose(np.array([self.m_paths_latlon_org,]*len(x_axis)),(1,2,3,0))
            
            l2_wm=np.sqrt((paths_wm_org_ext[0,:,:,:]-self.m_paths_wm_noisy[0,:,:,:])**2+(paths_wm_org_ext[1,:,:,:]-self.m_paths_wm_noisy[1,:,:,:])**2)
            l2_latlon=np.sqrt((paths_latlon_org_ext[0,:,:,:]-self.m_paths_latlon_noisy[0,:,:,:])**2+(paths_latlon_org_ext[1,:,:,:]-self.m_paths_latlon_noisy[1,:,:,:])**2)
            
            MSE_WM = np.mean(l2_wm,axis=(0,1))
            MSE_latlon = np.mean(l2_latlon,axis=(0,1))
            
            plt.plot(x_axis,MSE_WM,'b-*',x_axis,MSE_latlon,'r-*')
            
            # Plotting Longitude
            ax = plt.gca()
            ax.invert_xaxis()
            plt.yscale('log')
            #plt.xscale('log')
            plt.grid()
            plt.title('Mean square error')
            plt.xlabel('Noise level (meters)')
            plt.ylabel('MSE')
            plt.show()
      