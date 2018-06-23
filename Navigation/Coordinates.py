# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Coordinates conversions
 Project     : Simulation environment for BckTrk app
 File        : Coordinates.py
 -----------------------------------------------------------------------------

   Description :


   
   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   26-Mai-2018  1.0      Rami      File created
 =============================================================================
 
"""
import numpy as np
import pygeodesy as geo

## Bring the logger
import logging
logger = logging.getLogger('BckTrk')

def convert_wm_to_elatlon(arr) :
    logger.debug("Converts an array of webmercator points to ellipsoid Latitude and longitude coordinate")
    n       = arr.shape[0]
    temp    = np.empty(n, dtype=object)
    
    for i in range(n):
        temp[i] = arr[i].toLatLon(geo.ellipsoidalVincenty.LatLon, datum=geo.Datums.WGS84)
    return temp

def convert_elatlon_to_wm(arr) :
    logger.debug("Converts an array of ellipsoid Latitude and longitude coordinates to webmercator points")
    n       = arr.shape[0]
    temp    = np.empty(n, dtype=object)
    
    for i in range(n):
        temp[i] = geo.toWm(arr[i])
    return temp

def convert_xy_to_wm(arr_float) :
    logger.debug("Converts an array of x-y datapoints (2,n) array into webmercator points")
    n       = arr_float.shape[1]
    temp    = np.empty(n, dtype=object)
    
    for i in range(n):
        temp[i] = geo.Wm(arr_float[0,i],arr_float[1,i])
    return temp

   ## Generate longitute and lattitude equivalent data
def generate_latlon_array(arr_wm):
    logger.debug("Generating a path of LatLon coordinates")
    n               = arr_wm.shape[1]
    temp            = np.empty((2,n), dtype=object)
    positions_wm    = convert_xy_to_wm(arr_wm)
    latlon = convert_wm_to_elatlon(positions_wm)

    for i in range(n):
        temp[0,i] = latlon[i].lat
        temp[1,i] = latlon[i].lon
    return temp