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

def convert_wm_to_elatlon(arr) :
    """converts an array of webmercator points to ellipsoid Latitude and longitude coordinates"""
    n = arr.shape[0]
    temp = np.empty(n, dtype=object)
    
    for i in range(n-1):
        temp[i] = arr[i].toLatLon(geo.ellipsoidalVincenty.LatLon, datum=geo.Datums.WGS84)
    return temp

def convert_elatlon_to_wm(arr) :
    """converts an array of ellipsoid Latitude and longitude coordinates to webmercator points"""
    n = arr.shape[0]
    temp = np.empty(n, dtype=object)
    
    for i in range(n-1):
        temp[i] = geo.toWm(arr[i])
    return temp

def convert_xy_to_wm(arr_float) :
    """converts an array of x-y datapoints (2,n) array into webmercator points"""
    n = arr_float.shape[1]
    temp = np.empty(n, dtype=object)
    
    for i in range(n-1):
        temp[i] = geo.Wm(arr_float[0,i],arr_float[1,i])
    return temp