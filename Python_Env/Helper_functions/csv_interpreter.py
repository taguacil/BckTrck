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
# Import python libraries
import pandas as pd
import numpy as np
import json
import sys
import re

## Logging
import logging

logger = logging.getLogger('BckTrk')


def cmp(a, b):
    return (a > b) - (a < b)


def version_cmp(version1, version2):
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]

    return cmp(normalize(version1), normalize(version2))


def split_array(arr, l):
    size = int(arr.shape[0])
    overlap = int(l / 2)
    last = int((size / overlap) * overlap)
    clipped_array = arr[:last]

    blocks = []
    for i in range(int((size / overlap)) - 1):
        temp = clipped_array[i * overlap:(i * overlap + l)]
        blocks.append(temp)

    return np.array(blocks).T


def merge_arrays(lis, l):
    temp = lis[:int(3 * l / 4), 0]
    size = lis.shape[-1]

    for i in range(1, size - 1):
        temp = np.concatenate((temp, lis[int(l / 4):int(3 * l / 4), i]))

    if size > 1:
        temp = np.concatenate((temp, lis[int(l / 4):, -1]))

    return temp

def munge_csv(path, path_length):
    sheet = pd.read_csv(path)
    if version_cmp(pd.__version__, "0.22.0") > 0:
        sheet.Date = sheet.Date.astype('datetime64[ns]')
        sheet['Sampling_interval_seconds'] = sheet.Date.map(pd.datetime.timestamp).diff()
    else:
        sheet['Sampling_interval_seconds'] = 1
        logger.warning("CSV date not used due to old pandas version")

    lat = sheet[['Latitude']].values
    lon = sheet[['Longitude']].values
    acc = sheet[' horizontal accuracy'].values
    interval = sheet['Sampling_interval_seconds'].values

    number_of_realizations = int(acc.shape[0] / path_length)

    if number_of_realizations != 0:
        latlon = np.array([split_array(lat, int(path_length)), split_array(lon, int(path_length))])
        acc = split_array(acc, int(path_length))
        interval = split_array(interval, int(path_length))
        return latlon, acc, interval
    else:
        logger.error("Paths provided smaller than requested path length")
        sys.exit("Paths provided smaller than requested path length")


if __name__ == "__main__":
    numberOfArgument = len(sys.argv)
    if numberOfArgument != 3:
        print("Please specify path of CSV file to load and path of json output file")

    else:
        file = sys.argv[1]
        destination = sys.argv[2]
        latlon, acc, interval = munge_csv(file)
        output = {"latlon_data": [list(latlon[0]), list(latlon[1])], "accuracy": list(acc),
                  "sampling_interval_seconds": list(interval)}
        with open(destination, 'w') as fp:
            json.dump(output, fp)
