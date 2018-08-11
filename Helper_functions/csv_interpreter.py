# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Intrepreting CSV files
 Project     : Simulation environment for BckTrk app
 File        : cv_interpreter.py
 -----------------------------------------------------------------------------

   Description :

   This file contains functions that munge CSV files produced by the App to 
   capture real Lat Lon data, accuracy and sampling intervals. 
   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   25-Apr-2018  1.0      Rami      File created
 =============================================================================
 
"""
import numpy as np 
import pandas as pd 
import json
import sys

def munge_csv(path) :
    sheet = pd.read_csv(path)
    sheet.Date = sheet.Date.astype('datetime64[ns]')
    sheet['Sampling_interval_seconds'] = sheet.Date.map(pd.datetime.timestamp).diff()
    latlon = sheet[['Latitude','Longitude']].values.T
    acc = sheet[' horizontal accuracy'].values
    interval = sheet['Sampling_interval_seconds'].values
    
    return latlon, acc, interval 

if __name__ == "__main__" :
    numberOfArgument =  len(sys.argv)  
    if numberOfArgument !=3 :
        print("Please specify path of CSV file to load and path of json output file")
        
    else :
        file = sys.argv[1]
        destination = sys.argv[2]
        latlon, acc, interval = munge_csv(file)
        output = {"latlon_data" : [list(latlon[0]),list(latlon[1])], "accuracy":list(acc), "sampling_interval_seconds":list(interval)}
        with open(destination, 'w') as fp:
            json.dump(output, fp)