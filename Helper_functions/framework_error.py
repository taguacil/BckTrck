# -*- coding: utf-8 -*-
"""
 =============================================================================
 Title       : Error handling script
 Project     : Simulation environment for BckTrk app
 File        : framework_error.py
 -----------------------------------------------------------------------------

   Description :

   This file is responsible for all exception handling across the framework

   References :

   -
 -----------------------------------------------------------------------------
 Revisions   :
   Date         Version  Name      Description
   25-Sep-2018  1.0      Taimir    File created
 =============================================================================

"""
from enum import Enum

class CErrorTypes (Enum):
    value = 1
    type = 2
    ioerror = 3
    range = 4

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class CFrameworkError(Error):
    """Exception raised for errors in the framework.

    Attributes:
        Dictionary containing all caller error information
    """

    def __init__(self, callerdict):

        if "file" in callerdict:
            self.callerfile = callerdict["file"]
        else:
            self.callerfile = None

        if "message" in callerdict:
            self.callermessage = callerdict["message"]
        else:
            self.callermessage = None

        if "errorType" in callerdict:
            self.callertype = callerdict["errorType"]
        else:
            self.callertype = None
