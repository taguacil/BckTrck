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
import platform 

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')
    
if platform.system() == "Windows" :
    direc_ident = "\\"
else :
    direc_ident = "/"
    
## the main processing function 
def process_data (params):
    data_obj = cProcessFile (params)
    if params["CSV_DATA"]["bUse_csv_data"] :
        data_obj.plot_real_path_recon()
    else :
        _, _,_,_, MSE_r_latlon, reconstructed_db_latlon = data_obj.calculate_MSE()
        data_obj.plot_path_org_2d()
        data_obj.plot_path_noisy_2d()
        data_obj.plot_MSE()
        data_obj.analyze_DCT()
        params["RESULTS"]["reconstructed_db_latlon"] = reconstructed_db_latlon
        params["RESULTS"]["MSE_latlon"] = MSE_r_latlon
        if params["bSimplified_Results"] :
            del params['RESULTS']['paths_wm_org']
            del params['RESULTS']['paths_latlon_org']              
            del params['RESULTS']['paths_wm_real']               
            del params['RESULTS']['paths_latlon_real']           
            del params['RESULTS']['paths_wm_noisy']                 
            del params['RESULTS']['paths_latlon_noisy']             
            del params['RESULTS']['transformed_paths']             
            del params['RESULTS']['transformed_real_paths']          
            del params['RESULTS']['reconstructed_latlon_paths']     
            del params['RESULTS']['reconstructed_WM_paths']          
            del params['RESULTS']['reconstructed_real_latlon_paths'] 
            del params['RESULTS']['reconstructed_real_WM_paths']     
            
    data_obj.set_pickle_file(params)

class cProcessFile:
    ## Constructor
    def __init__(self,struct):
        self.m_plotStruct           = struct['PLOT']
        
        if struct['bUse_filename'] :
            self.m_filename         = struct['workingDir'] + direc_ident + 'Results' + direc_ident + 'BckTrk_Res_' + struct['currentTime'].strftime("%Y%m%d_%H-%M-%S") + '_' + struct['filename'] +'.txt'
        else :
            self.m_filename         = struct['workingDir'] + direc_ident + 'Results' + direc_ident + 'BckTrk_Res_' + struct['currentTime'].strftime("%Y%m%d_%H-%M-%S")+'.txt' 
            
        self.m_acquisition_length   = struct['gps_freq_Hz']*struct['acquisition_time_sec']
        self.m_number_realization   = struct ['realization']
        
        self.m_path_length          = struct['CSV_DATA']['path_length']
        
        self.m_noise_level_meter    = struct['noise_level_meter']
        
        self.m_bReconstruct         = struct['bReconstruct']
        self.m_lasso_sampling_ratio = struct['RCT_ALG_LASSO']['sampling_ratio']
    
        self.m_paths_wm_org                 = struct['RESULTS']['paths_wm_org']
        self.m_paths_latlon_org             = struct['RESULTS']['paths_latlon_org']
        self.m_paths_wm_real                = struct['RESULTS']['paths_wm_real']
        self.m_paths_latlon_real            = struct['RESULTS']['paths_latlon_real']
        self.m_paths_wm_noisy               = struct['RESULTS']['paths_wm_noisy']
        self.m_paths_latlon_noisy           = struct['RESULTS']['paths_latlon_noisy']
        self.transformed_paths              = struct['RESULTS']['transformed_paths']
        self.transformed_real_paths         = struct['RESULTS']['transformed_real_paths']
        self.reconstructed_latlon_paths     = struct['RESULTS']['reconstructed_latlon_paths']
        self.reconstructed_wm_paths         = struct['RESULTS']['reconstructed_WM_paths']
        self.reconstructed_real_latlon_paths     = struct['RESULTS']['reconstructed_real_latlon_paths']
        self.reconstructed_real_wm_paths         = struct['RESULTS']['reconstructed_real_WM_paths']
        
        
    ## Write dictionary into pickle format to txt file    
    def set_pickle_file(self,struct) :
        with open (self.m_filename, 'wb') as txt_file:
            pickle.dump(struct, txt_file)
            
    ## Read from txt file pickle format into dictionary        
    def get_pickle_file(self):
        with open (self.m_filename, 'rb' ) as txt_file_read :
            return pickle.load (txt_file_read)
    
    def calculate_MSE(self):
        x_axis = self.m_noise_level_meter
        paths_wm_org_ext=np.transpose(np.array([self.m_paths_wm_org,]*len(x_axis)),(1,2,3,0))
        paths_latlon_org_ext=np.transpose(np.array([self.m_paths_latlon_org,]*len(x_axis)),(1,2,3,0))
            
        l2_noise_wm=np.sqrt(np.mean((paths_wm_org_ext[0,:,:,:]-self.m_paths_wm_noisy[0,:,:,:])**2+(paths_wm_org_ext[1,:,:,:]-self.m_paths_wm_noisy[1,:,:,:])**2,axis=0))
        l2_noise_latlon=np.sqrt(np.mean((paths_latlon_org_ext[0,:,:,:]-self.m_paths_latlon_noisy[0,:,:,:])**2+(paths_latlon_org_ext[1,:,:,:]-self.m_paths_latlon_noisy[1,:,:,:])**2,axis=0))
            
        MSE_noise_WM = np.mean(l2_noise_wm,axis=0)
        MSE_noise_latlon = np.mean(l2_noise_latlon,axis=0)
        MSE_r_latlon = {}    
        MSE_r_wm = {}
        max_error = {}
        reconstructed_db_latlon = {}
            
        if self.m_bReconstruct:
            logger.info('Calculating MSE of reconstructed paths latlon')
                
            for key in self.reconstructed_latlon_paths.keys():
                r_path = self.reconstructed_latlon_paths[key]
                    
                l2_r_latlon=np.sqrt(np.mean((paths_latlon_org_ext[0,:,:,:]-r_path[0,:,:,:])**2+(paths_latlon_org_ext[1,:,:,:]-r_path[1,:,:,:])**2,axis=0))
                MSE_r_latlon[key] = np.mean(l2_r_latlon,axis=0)
                reconstructed_db_latlon[key] = 20*np.log10(MSE_r_latlon[key]/MSE_noise_latlon) # Ratio of reconstruction error to error from noisy data in decibels
                
                r2_path = self.reconstructed_wm_paths[key]
                    
                l2_r_wm   = np.sqrt(np.mean((paths_wm_org_ext[0,:,:,:]-r2_path[0,:,:,:])**2+(paths_wm_org_ext[1,:,:,:]-r2_path[1,:,:,:])**2,axis=0))
                MSE_r_wm[key]  = np.mean(l2_r_wm,axis=0)
                max_error[key] = np.mean(np.sqrt(np.max((paths_wm_org_ext[0,:,:,:]-r2_path[0,:,:,:])**2+(paths_wm_org_ext[1,:,:,:]-r2_path[1,:,:,:])**2,axis=0)),axis=0)
            
        return MSE_noise_WM, MSE_noise_latlon, MSE_r_wm, max_error, MSE_r_latlon, reconstructed_db_latlon
        
            
    ## Plot real path and reconstruction 
    def plot_real_path_recon(self):
        logger.info('Plotting real path and stitched reconstruction')
        latlon_real = self.m_paths_latlon_real.reshape((2, self.m_path_length*self.m_number_realization))
                
        plt.plot(latlon_real[1],latlon_real[0],'b-*')
        if self.m_bReconstruct:
                logger.info('Plotting MSE of reconstructed paths latlon')
                
                for key in self.reconstructed_real_latlon_paths.keys():
                    r_path = self.reconstructed_real_latlon_paths[key].reshape((2, self.m_path_length*self.m_number_realization))
                    
                    l2_r_latlon=np.sqrt(np.mean((latlon_real[0]-r_path[0])**2+(latlon_real[1]-r_path[1])**2))

                    plt.plot(r_path[1],r_path[0],'r-*')
                    print('L2 for %s is %.6f'%(key, l2_r_latlon))
        
        plt.grid()
        
        plt.title('Real path from CSV data broken into %d segments'%(self.m_number_realization))
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend(['Real path', 'Reconstruction'])
        plt.show()
    
    ## Original path in 2D plotting
    def plot_path_org_2d(self) : 
        #Set of params
        x_axis = range(0, self.m_acquisition_length)

        ####################
        if self.m_plotStruct['bPlotPath_WM_org']:
            logger.info ('Plotting original path in webmercator coordinates')
            if self.m_plotStruct['bPlotAllrealizations']:
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
        if self.m_plotStruct['bPlotWM_time_org']:
            logger.info ('Plotting original path in webmercator coordinates')
            if self.m_plotStruct['bPlotAllrealizations']:
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
            if self.m_plotStruct['bPlotAllrealizations']:
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
        if self.m_plotStruct['bPlotLonLat_time_org']:
            logger.info ('Plotting original longitude and latitude')
            if self.m_plotStruct['bPlotAllrealizations']:
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
            if self.m_plotStruct['bPlotAllrealizations']:
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
        noise_level = self.m_noise_level_meter
        
        for noise in range(len(noise_level)):
            paths_wm_noisy      = self.m_paths_wm_noisy[:,:,:,noise]
            paths_wm_org        = self.m_paths_wm_org
            paths_latlon_noisy  = self.m_paths_latlon_noisy[:,:,:,noise]
            paths_latlon_org    = self.m_paths_latlon_org
            
            if self.m_plotStruct['bPlotPath_WM_noisy']:
                logger.info ('Plotting noisy path in webmercator coordinates')
                if self.m_plotStruct['bPlotAllrealizations']:
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
            if self.m_plotStruct['bPlotPath_LonLat_noisy']:
                logger.info ('Plotting noisy path in lonlat coordinates')
                if self.m_plotStruct['bPlotAllrealizations']:
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
            if self.m_plotStruct['bPlotWM_time_noisy']:
                logger.info ('Plotting noisy path in webmercator coordinates')
                if self.m_plotStruct['bPlotAllrealizations']:
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
            
                if self.m_plotStruct['bPlotAllrealizations']:
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
            if self.m_plotStruct['bPlotLonLat_time_noisy']:
                logger.info ('Plotting noisy longitude and latitude')
                if self.m_plotStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
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
                
                if self.m_plotStruct['bPlotAllrealizations']:
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
           
             ####################
            if self.m_plotStruct['bPlotLonLat_time_reconst']:
                logger.info ('Plotting reconstructed path in comparison to original')
                if self.m_plotStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis, paths_latlon_org[0,:,k],'-*',label="Original latitude for realization %.1f"%(k))
                        
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_latlon_paths.keys():
                            r_path = self.reconstructed_latlon_paths[key]
                            plt.plot(x_axis,r_path[0,:,k,noise],'-*',label="Latitude for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                        
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    plt.plot(x_axis, paths_latlon_org[0,:,0],'-*',label="Original latitude")
                    
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_latlon_paths.keys():
                            r_path = self.reconstructed_latlon_paths[key]
                            plt.plot(x_axis,r_path[0,:,0,noise],'-*',label="Latitude for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                      
                    
                #Plotting Latitude
                buf = "Noisy latitude for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.legend(loc="upper right")
                plt.xlabel('Number of steps')
                plt.ylabel('Latitude')
                plt.show() 
                
                if self.m_plotStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis,paths_latlon_org[1,:,k],'-*',label="Original longitude for realization %.1f"%(k))
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_latlon_paths.keys():
                            r_path = self.reconstructed_latlon_paths[key]
                            plt.plot(x_axis,r_path[1,:,0,noise],'-*',label="Longitude for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                          
                else:
                    plt.plot(x_axis,paths_latlon_org[1,:,0],'-*',label="Original longitude")
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_latlon_paths.keys():
                            r_path = self.reconstructed_latlon_paths[key]
                            plt.plot(x_axis,r_path[1,:,0,noise],'-*',label="Longitude for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                      
            
                #Plotting Longitude
                buf = "Noisy longitude for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.legend(loc="upper right")
                plt.xlabel('Number of steps')
                plt.ylabel('Longitude')
                plt.show()
                
            ####################
            if self.m_plotStruct['bPlotWM_time_reconst']:
                logger.info ('Plotting reconstructed path in WM in comparison to original')
                if self.m_plotStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis, paths_wm_org[0,:,k],'-*',label="Original x for realization %.1f"%(k))
                        
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_wm_paths.keys():
                            r_path = self.reconstructed_wm_paths[key]
                            plt.plot(x_axis,r_path[0,:,k,noise],'-*',label="x for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                        
                else:
                    logger.warning ('Plotting only first realization for visibility')
                    plt.plot(x_axis, paths_wm_org[0,:,0],'-*',label="Original x")
                    
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_wm_paths.keys():
                            r_path = self.reconstructed_wm_paths[key]
                            plt.plot(x_axis,r_path[0,:,0,noise],'-*',label="x for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                      
                    
                #Plotting Latitude
                buf = "Noisy x for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.legend(loc="upper right")
                plt.xlabel('Number of steps')
                plt.ylabel('x')
                plt.show() 
                
                if self.m_plotStruct['bPlotAllrealizations']:
                    for k in range (self.m_number_realization):
                        plt.plot(x_axis,paths_wm_org[1,:,k],'-*',label="Original y for realization %.1f"%(k))
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_wm_paths.keys():
                            r_path = self.reconstructed_wm_paths[key]
                            plt.plot(x_axis,r_path[1,:,0,noise],'-*',label="y for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                          
                else:
                    plt.plot(x_axis,paths_wm_org[1,:,0],'-*',label="Original y")
                    if self.m_bReconstruct:
                        logger.info('Plotting MSE of reconstructed paths')
                
                        for key in self.reconstructed_wm_paths.keys():
                            r_path = self.reconstructed_wm_paths[key]
                            plt.plot(x_axis,r_path[1,:,0,noise],'-*',label="y for %s with %.1f %% sampling ratio"%(key, self.m_lasso_sampling_ratio*100  ))
                      
            
                #Plotting Longitude
                buf = "Noisy y for noise level %d (meters)" % (noise_level[noise])
                plt.grid()
                plt.title(buf)
                plt.legend(loc="upper right")
                plt.xlabel('Number of steps')
                plt.ylabel('y')
                plt.show()
                
    ## Plot MSE - mean error rate
    def plot_MSE(self) : 
        if self.m_plotStruct['bPlotMSE']: 
            logger.info ('Plotting MSE of WM and latlon values')
            #Set of params
            x_axis = self.m_noise_level_meter
            MSE_noise_WM, MSE_noise_latlon, MSE_r_wm, max_error, MSE_r_latlon,_ = self.calculate_MSE()
            
            if self.m_bReconstruct:
                logger.info('Plotting MSE of reconstructed paths latlon')
                for key in self.reconstructed_latlon_paths.keys():
                    plt.plot(x_axis,MSE_r_latlon[key],'-*',label="MSE_latlon for %s with %.1f %% SR"%(key, self.m_lasso_sampling_ratio*100  ))
                    
            
            # Plotting MSE
            plt.plot(x_axis,MSE_noise_latlon,'-*',label="MSE_latlon")
            ax = plt.gca()
            ax.invert_xaxis()
            plt.yscale('log')
            plt.xscale('log')
            plt.grid()
            plt.legend(loc="upper right")
            plt.title('Mean square error for %d samples and %d iteratirons'%(self.m_acquisition_length, self.m_number_realization))
            plt.xlabel('Noise level (meters)')
            plt.ylabel('MSE')
            plt.show()
            
            if self.m_bReconstruct:
                logger.info('Plotting MSE of reconstructed paths WM')
                
                for key in self.reconstructed_wm_paths.keys():
                    plt.plot(x_axis,MSE_r_wm[key],'-*',label="MSE_WM for %s with %.1f %% SR"%(key, self.m_lasso_sampling_ratio*100))
                    plt.plot(x_axis,max_error[key],'-x',label="Average Max Error for %s with %.1f %% SR"%(key, self.m_lasso_sampling_ratio*100))
       
            plt.plot(x_axis,MSE_noise_WM,'-*',label="MSE_WM")
            ax = plt.gca()
            ax.invert_xaxis()
            plt.yscale('log')
            plt.xscale('log')
            plt.grid()
            plt.legend(loc="upper right")
            plt.title('Mean square error for %d samples and %d iteratirons'%(self.m_acquisition_length, self.m_number_realization))
            plt.xlabel('Noise level (meters)')
            plt.ylabel('MSE')
            plt.show()
            
            
    
    ## DCT analysis
    def analyze_DCT(self) : 
        if self.m_plotStruct['bPlotDCTAnalysis']:
            logger.info ('Triggering DCT analysis for latlon coordinates')
            x_axis = self.m_noise_level_meter
            transformed_paths=self.transformed_paths
            percentiles = [0.05,0.01,0.005]
            
            percent_len=len(percentiles)
            noise_level_len=len(x_axis)
            
            average_lat = np.zeros((percent_len,noise_level_len))
            average_lon = np.zeros((percent_len,noise_level_len))
            
            for per in range(percent_len):
                value = (np.abs(transformed_paths[0,:,:,:])/np.abs(transformed_paths[0,0,:,:])<percentiles[per])
                average_lat[per,:] = np.mean(value, axis=(0,1))
                average_lon[per,:] = np.mean((np.abs(self.transformed_paths[1,:,:,:])/np.abs(self.transformed_paths[1,0,:,:])<percentiles[per]), axis=(0,1))
            
            for i in range(percent_len):
                plt.plot(x_axis,average_lat[i,:]*100,"-*",label="%.2f %%"%(percentiles[i]*100))
            
            # Plotting percentages wrt. noise levels for lattitude
            #ax = plt.gca()
            #ax.invert_xaxis()
            #plt.yscale('log')
            plt.xscale('log')
            plt.grid()
            plt.legend(loc="upper right")
            plt.title('DCT analysis for lattitude')
            plt.xlabel('Noise level (meters)')
            plt.ylabel('Percentage')
            plt.show()

            for i in range(percent_len) :
                plt.plot(x_axis,average_lon[i,:]*100,"-*",label="%.2f %%"%(percentiles[i]*100))
            
            # Plotting percentages wrt. noise levels for lattitude
            #ax = plt.gca()
            #ax.invert_xaxis()
            #plt.yscale('log')
            plt.xscale('log')
            plt.grid()
            plt.legend(loc="upper right")
            plt.title('DCT analysis for longitute')
            plt.xlabel('Noise level (meters)')
            plt.ylabel('Percentage')
            plt.show()

