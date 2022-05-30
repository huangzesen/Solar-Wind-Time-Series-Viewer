# %matplotlib tk
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec

import numpy as np
from scipy.optimize import curve_fit
import gc
import time
from scipy.signal import savgol_filter
import pandas as pd
from numba import jit,njit,prange,objmode 
import os
from pathlib import Path
from glob import glob
import pickle

from datetime import datetime

# SPDF API
# from cdasws import CdasWs
# cdas = CdasWs()

au_to_km = 1.496e8  # Conversion factor
rsun     = 696340   # Sun radius in units of  [km]

from scipy import constants
psp_ref_point   =  0.06176464216946593 # in [au]
mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu_0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
m_p             =  constants.m_p   # Proton mass [kg]
k          = constants.k                  # Boltzman's constant     [j/K]
au_to_km   = 1.496e8
T_to_Gauss = 1e4

from TSUtilities import resample_timeseries_estimate_gaps, TracePSD

@jit(nopython=True, parallel=True)      
def smoothing_function(x,y, window=2, pad = 1):
    xoutmid = np.zeros(len(x))*np.nan
    xoutmean = np.zeros(len(x))*np.nan
    yout = np.zeros(len(x))*np.nan
    for i in prange(len(x)):
        x0 = x[i]
        xf = window*x[i]
        
        if xf < np.max(x):
            e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
            if e<len(x):
                yout[i] = np.nanmean(y[i:e])
                xoutmid[i] = x[i] + np.log10(0.5) * (x[i] - x[e])
                xoutmean[i] = np.nanmean(x[i:e])
    return xoutmid, xoutmean, yout 


class CalculatePSD:

    def __init__(
        self, path):
        self.path = path


    def LoadCase(self):
        try:
            case = pd.read_pickle(self.path)
            self.case = case
            self.sc = case['sc']
            self.start_time = case['start_time']
            self.end_time = case['end_time']
        except:
            raise ValueError("Load pickle failed!")

    
    def PrepareData(self, resolution = 0.1):
        if self.sc == 0:
            dfraw = self.case['spdf_data']['spdf_infos']['mag_sc']['dataframe']
        elif self.sc == 1:
            dfraw = self.case['spdf_data']['spdf_infos']['mag_srf']['dataframe']
        else:
            raise ValueError("")
        df_resampled, fraction_missing, total_large_gaps, total_gaps = resample_timeseries_estimate_gaps(
            dfraw, resolution = 1000*resolution
        )
        self.resample_info = {
            'fraction_missing': fraction_missing,
            'total_large_gaps': total_large_gaps,
            'total_gaps': total_gaps
        }

        # Interpolate
        df_resampled = df_resampled.interpolate('linear')
        self.mag_resampled = df_resampled
        self.dt = resolution

    
    def CalculatePSD(self):
        x, y, z = self.mag_resampled['Bx'], self.mag_resampled['By'], self.mag_resampled['Bz']
        freqs, B_pow = TracePSD(x, y, z, self.dt)
        self.freqs = freqs
        self.B_pow = B_pow
    
    def SmoothPSD(self):

        _, xoutmean, yout = smoothing_function(self.freqs, self.B_pow)

        self.freqs_sm = xoutmean
        self.B_pow_sm = yout

    def StoreInfo(self):
        self.Diagnostics = {
            'sc': self.sc,
            'start_time': self.start_time,
            'end_time': self.end_time
        }

        self.Diagnostics['PSD'] = {
            'freqs': self.freqs,
            'PSD': self.B_pow,
            'sm_freqs': self.freqs_sm,
            'sm_PSD': self.B_pow_sm,
            'resample_info': self.resample_info
        }

    def SaveCase(self, spath = None, sname = None, no_replace = True):
        """
        Save the PSD data
        default save path would be the ../Diagnostics/Diagnostics.pkl
        """
        if spath is None:
            spath = Path(self.path).absolute().parent.parent.joinpath("Diagnostics")
        if sname is None:
            sname = "Diagnostics.pkl"

        savpath = spath.joinpath(sname)

        if no_replace:
            if os.path.exists(savpath):
                savpath = spath.joinpath(Path(sname).stem+"_new.pkl")

        print("Saving to... %s"%(str(savpath)))
        pickle.dump(self.Diagnostics, open( savpath, "wb" ) )

    def ProducePSD(self,spath = None, sname = None, no_replace = True):
        """ Wrapper, use after initialization """
        self.LoadCase()
        self.PrepareData()
        self.CalculatePSD()
        self.SmoothPSD()
        self.StoreInfo()
        self.SaveCase(spath = spath, sname = sname, no_replace = no_replace)
