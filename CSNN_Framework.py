# %-*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Compressive sensing framework
 Project     : Simulation environment for BckTrk app
 File        : CSNN_sim.py
 -----------------------------------------------------------------------------

   Description :

   This file contains the main function which is the core of the simulation.
   More details here  
   
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
import sys
import os
import json
import datetime
import logging

## User-defined library import
from Helper_functions.proc_results import process_data 

## Parameters / Config files handling
workingDir = os.getcwd()
paramPath = workingDir + '\\Parameter_files\\'
local_struct = json.load(open(paramPath + 'default_config.json', 'r'))

try :
    os.stat (workingDir+'\\Logs\\')
except : 
    os.mkdir(workingDir+'\\Logs\\')
        
try :
    os.stat (workingDir+'\\Results\\')
except :
    os.mkdir(workingDir+'\\Results\\')
     
# create logger with 'spam_application'
logger = logging.getLogger('BckTrk')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
now = datetime.datetime.now()
fh = logging.FileHandler(workingDir + '\\Logs\\' + 'BckTrk_Log_' + now.strftime("%Y-%m-%d")+'.log')
fh.setLevel(logging.DEBUG)

local_struct['currentTime']=now
local_struct['workingDir'] = workingDir

# create console handler with same log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

## Define exit function

# TODO Since an exception can occur in one of the sub modules, this function will never be called 
# Doubling logs bug will happen, solution 1 make all function return a value to main
# pass the exit function somehow to all submodules 

def exit_framework():
     
    fh.close()
    ch.close()
    logger.removeHandler(fh)
    logger.removeHandler(ch)
    sys.exit(0)

## Business logic for input arguments to main function 
numberOfArgument =  len(sys.argv)  
if numberOfArgument == 1 :
    logger.info ('Default config will be used')
    
elif numberOfArgument == 2 :
    parameters_filename = sys.argv[1] #first argument should be the parameters filename
    updated_config = json.load(open( paramPath + parameters_filename, 'r')) #JSON loads a dictionary written in a JSON file as a python dictionary
    for key in updated_config :
        local_struct[key] = updated_config[key]
else :
    logger.error ('Invalid number of arguments %d' %numberOfArgument)
    exit_framework()
    

## Main function definition 
def main() :
            
    ##Generate random data
    x = np.random.randint(0,100,size=(2,20))
    
    #Store data in local struct
    local_struct['RESULTS']['data_2d'] = x
    process_data(local_struct)
    exit_framework()
    
## Main function definition  MUST BE at the END OF FILE    
if __name__ == "__main__" :
    main()
    
