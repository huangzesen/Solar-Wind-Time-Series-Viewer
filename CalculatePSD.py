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

# from TSUtilities import resample_timeseries_estimate_gaps, TracePSD

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

def TracePSD(x,y,z,dt):
    """ 
    Estimate Power spectral density:

    Inputs:

    u : timeseries, np.array
    dt: 1/sampling frequency

    """
    
    B_pow = np.abs(np.fft.rfft(x, norm='ortho'))**2 \
          + np.abs(np.fft.rfft(y, norm='ortho'))**2 \
          + np.abs(np.fft.rfft(z, norm='ortho'))**2

    freqs = np.fft.fftfreq(len(x), dt)
    freqs = freqs[freqs>0]
    idx   = np.argsort(freqs)
    
    return freqs[idx], B_pow[idx]



def resample_timeseries_estimate_gaps(df, resolution = 100, large_gaps = 10):
    """
    Resample timeseries and estimate gaps, default setting is for FIELDS data
    Resample to 10Hz and return the resampled timeseries and gaps infos
    Input: 
        df: input time series
    Keywords:
        resolution  =   100 [ms]    ## Select resolution [ms]
        large_gaps  =   10 [s]      ## large gaps in timeseries [s]
    Outputs:
        df_resampled        :       resampled dataframe
        fraction_missing    :       fraction of missing values in the interval
        total_large_gaps    :       fraction of large gaps in the interval
        total_gaps          :       total fraction of gaps in the interval   
    """
    # Find first col of df
    first_col = df.columns[0]
    
    # estimate duration of interval selected in seconds #
    interval_dur =  (df.index[-1] - df.index[0])/np.timedelta64(1, 's')

    # Resample time-series to desired resolution # 
    df_resampled = df.resample(str(int(resolution))+"ms").mean()

    # Estimate fraction of missing values within interval #
    fraction_missing = 100 *(df_resampled[first_col].isna().sum()/ len(df_resampled))

    # Estimate sum of gaps greater than large_gaps seconds
    res          = (df_resampled.dropna().index.to_series().diff()/np.timedelta64(1, 's'))
    
    
     # Gives you the fraction of  large gaps in timeseries 
    total_large_gaps   = 100*( res[res.values>large_gaps].sum()/ interval_dur  ) 
    
    # Gives you the total fraction  gaps in timeseries
    total_gaps         =  100*(res[res.values>int(resolution)*1e-3].sum()/ interval_dur  )    
    
    return df_resampled, fraction_missing, total_large_gaps, total_gaps




class CalculatePSD:

    def __init__(
        self, path):
        """
        Initialize the class with fig, ax, and plot data
        data: {'xdata':, 'ydata':}
        """
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
        if self.case['Diagnostics'] == None:
            self.case['Diagnostics'] = {}
        else:
            pass

        self.case['Diagnostics']['PSD'] = {
            'freqs': self.freqs,
            'PSD': self.B_pow,
            'sm_freqs': self.freqs_sm,
            'sm_PSD': self.B_pow_sm,
            'resample_info': self.resample_info
        }

    def SaveCase(self):
        parent = Path(self.path).absolute().parent
        stem = Path(self.path).absolute().stem
        f = parent.joinpath("PSD").joinpath(stem+"_PSD.pkl")
        pickle.dump(self.case['Diagnostics']['PSD'], open( f, "wb" ) )

