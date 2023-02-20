# %matplotlib tk
from distutils.log import warn
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
from gc import collect

import datetime
from numba import jit,njit,prange
import numba

# SPDF API
from cdasws import CdasWs
cdas = CdasWs()

# SPEDAS API
# make sure to use the local spedas
import sys
sys.path.insert(0,"../pyspedas")
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data
from scipy import signal
# import TurbPy as turb

au_to_km = 1.496e8  # Conversion factor
rsun     = 696340   # Sun radius in units of  [km]

from scipy import constants
psp_ref_point   =  0.06176464216946593 # in [au]
mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu_0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
m_p             =  constants.m_p   # Proton mass [kg]
k          = constants.k                  # Boltzman's constant     [j/K]
au_to_km   = 1.496e8
au_to_rsun = 215.032
T_to_Gauss = 1e4

from LoadData import LoadHighResMagWrapper
from StructureFunctions import MagStrucFunc


# -----------  Tools ----------- #


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


def SolarWindCorrelationLength(R, Vsw0 = 348.48, Vsw = None, correction = False):
    """ 
    Calculate the Recommended Interval length based on correlation length 
    Reference: 	arXiv:2205.00526

    The correlation length scale as:
    R < 0.3 AU          : R^{0.97 +- 0.04}
    R ~ [0.3AU, 1.0AU]  : R^{0.29 +- 0.01}
    R > 1 AU            : R^{0.27 +- 0.01}

    For simplicity, the value we adapt are:
    R < 0.3 AU          : R^{1.0}
    R ~ [0.3AU, 1.0AU]  : R^{0.3}
    R > 1 AU            : R^{0.27}

    The reference point is arbitrarily fixed to be T = 24Hr at R = 0.3AU,
    hence the interval length scales approximately as:
    T = 12Hr, R = 0.1AU
    T = 24Hr, R = 0.3AU
    T = 34Hr, R = 1.0AU

    Option: Correction

    However, different solar interval have different advection time, 
    it would be more reasonable to rescale the radial distance with
    a reference solar wind speed. In this function, this speed is chosen
    to be the averaged solar wind speed observed by PSP, i.e.
    Vsw0 = 348.48 [km/s]

    The radial distance would be therefore corrected as:
    Rnew = R * Vsw0/Vsw

    Input:      
        R [au]                  Distance to the sun in Astronomical Unit
    Keywords: 
        Vsw0 = 348.48 [km/s]    The averaged solar wind speed observed by PSP
        Vsw  = None [km/s]      The real solar wind speed provided 
        correction = False      Do you want to do the correction?       
    Output:     
        T [Hr]                  The recommended interval length
    """

    # correct R with provided solar wind speed
    if correction:
        R = R/Vsw * Vsw0
    else:
        pass

    # calculate the recommend interval length
    # if R < 0.3:
    #     T = (R/0.3) * 24
    # elif ((R >= 0.3) & (R < 1.0)):
    #     T = (R/0.3)**0.3 * 24
    # else:
    #     T = (1.0/0.3)**0.3 * (R/1.0)**0.27 * 24

    # simple geometric growth
    T = (R/0.3)*24

    return T



@jit(nopython=True, parallel=True)
def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices
    


@jit(nopython=True, parallel=True)      
def smoothing_function(x,y, window=2, pad = 1):
    """
    smoothing function
    input: x, y   numpy array
    output: xmid, xmean, smoothed y
    """
    def bisection(array,value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if (value < array[0]):
            return -1
        elif (value > array[n-1]):
            return n
        jl = 0# Initialize lower
        ju = n-1# and upper limits.
        while (ju-jl > 1):# If we are not yet done,
            jm=(ju+jl) >> 1# compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl=jm# and replace either the lower limit
            else:
                ju=jm# or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):# edge cases at bottom
            return 0
        elif (value == array[n-1]):# and top
            return n-1
        else:
            return jl

    len_x    = len(x)
    max_x    = np.max(x)
    xoutmid  = np.full(len_x, np.nan)
    xoutmean = np.full(len_x, np.nan)
    yout     = np.full(len_x, np.nan)
    
    for i in prange(len_x):
        x0 = x[i]
        xf = window*x0
        
        if xf < max_x:
            #e = np.where(x  == x[np.abs(x - xf).argmin()])[0][0]
            e = bisection(x,xf)
            if e<len_x:
                yout[i]     = np.nanmean(y[i:e])
                xoutmid[i]  = x0 + np.log10(0.5) * (x0 - x[e])
                xoutmean[i] = np.nanmean(x[i:e])

    return xoutmid, xoutmean,  yout


def TracePSD(x,y,z,dt, norm = None):
    """ 
    Estimate Power spectral density:

    Inputs:

    u : timeseries, np.array
    dt: 1/sampling frequency

    norm = 'forward'
    can be {“backward”, “ortho”, “forward”}
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html

    'forward' method can keep the transformed spectrum comparable with different sampling frequency. 'ortho' will conserve energy between real space and fourier transformed space.

    """
    
    B_pow = np.abs(np.fft.rfft(x, norm=norm))**2 \
          + np.abs(np.fft.rfft(y, norm=norm))**2 \
          + np.abs(np.fft.rfft(z, norm=norm))**2

    freqs = np.fft.rfftfreq(len(x), dt)

    coeff = len(x)/(2*dt)
    
    return freqs, B_pow/coeff


def curve_fit_log_wrap(x, y, x0, xf):  
    # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        # s = np.min([s,e])
        # e = np.max([s,e])
        if s>e:
            s,e = e,s
        else:
            pass

        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = curve_fit_log(x[s:e],y[s:e])
            #print(fit)

            return fit, s, e, False
        else:
            return [np.nan, np.nan, np.nan],np.nan,np.nan, True


def linlaw(x, a, b) : 
    return a + x * b


def curve_fit_log(xdata, ydata) : 
    
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)


# ----------- Wavelet ------------ #
import pycwt as wavelet
mother_wave_dict = {
    'gaussian': wavelet.DOG(),
    'paul': wavelet.Paul(),
    'mexican_hat': wavelet.MexicanHat()
}

@numba.njit(nogil=True)
def norm_factor_Gauss_window(scales, dt):
   
    s             = scales*dt
    numer         = np.arange(-3*s, 3*s+dt, dt)
    multiplic_fac = np.exp(-(numer)**2/(2*s**2))
    norm_factor   = np.sum(multiplic_fac)
    window        = len(multiplic_fac)
   
    return window,  multiplic_fac, norm_factor

# def estimate_wavelet_coeff(df_b, dj ):
   
#     """
#     Method to calculate the  1) wavelet coefficients in RTN 2) The scale dependent angle between Vsw and Β
#     Parameters
#     ----------
#     df_b: dataframe
#         Magnetic field timeseries dataframe

#     mother_wave: str
#         The main waveform to transform data.
#         Available waves are:
#         'gaussian':
#         'paul': apply lomb method to compute PSD
#         'mexican_hat':
#     Returns
#     -------
#     freq : list
#         Frequency of the corresponding psd points.
#     psd : list
#         Power Spectral Density of the signal.
#     """
   
#     # Turn columns of df into arrays  
#     Br, Bt, Bn                           =  df_b.Br.values, df_b.Bt.values, df_b.Bn.values

#     # Estimate magnitude of magnetic field
#     mag_orig                             =  np.sqrt(Br**2 + Bt**2 +  Bn**2 )
   
#     # Estimate the magnitude of V vector
#     mag_v = np.sqrt(df_b['Br']**2).values

#     #Estimate sampling time of timeseries
#     dt                                   =  (df_b.dropna().index.to_series().diff()/np.timedelta64(1, 's')).median()

#     angles   = pd.DataFrame()
#     VBangles = pd.DataFrame()

#     # Estimate PSDand scale dependent fluctuations
#     db_x, db_y, db_z, freqs, PSD, scales = turb.trace_PSD_wavelet(Br, Bt, Bn, dt, dj,  mother_wave='morlet')


#     for ii in range(len(scales)):
#         #if np.mod(ii, 2)==0:
#            # print('Progress', 100*(ii/len(scales)))
#         try:
#             window, multiplic_fac, norm_factor= norm_factor_Gauss_window(scales[ii], dt)


#             # Estimate scale dependent background magnetic field using a Gaussian averaging window

#             res2_Br = (1/norm_factor)*signal.convolve(Br, multiplic_fac[::-1], 'same')
#             res2_Bt = (1/norm_factor)*signal.convolve(Bt, multiplic_fac[::-1], 'same')
#             res2_Bn = (1/norm_factor)*signal.convolve(Bn, multiplic_fac[::-1], 'same')


#             # Estimate magnitude of scale dependent background
#             mag_bac = np.sqrt(res2_Br**2 + res2_Bt**2 + res2_Bn**2 )

#             # Estimate angle
#             angles[str(ii+1)] = np.arccos(res2_Br/mag_bac) * 180 / np.pi

#             # Estimate VB angle
#             VBangles[str(ii+1)] = np.arccos((df_b['Br']*res2_Br)/(mag_bac*mag_v)) * 180 / np.pi

#             # Restric to 0< Θvb <90
#             VBangles[str(ii+1)][VBangles[str(ii+1)]>90] = 180 - VBangles[str(ii+1)][VBangles[str(ii+1)]>90]

#             # Restric to 0< Θvb <90
#             angles[str(ii+1)][angles[str(ii+1)]>90] = 180 - angles[str(ii+1)][angles[str(ii+1)]>90]
#         except:
#              pass

#     return db_x, db_y, db_z, angles, VBangles, freqs, PSD, scales

from numba import prange

@jit( parallel =True)
def estimate_PSD_wavelets_all_intervals(db_x, db_y, db_z, angles, freqs,   dt,  per_thresh, par_thresh):

    PSD_par = np.zeros(len(freqs))
    PSD_per = np.zeros(len(freqs))

    for i in range(np.shape(angles)[1]):

        index_per = (np.where(angles.T[i]>per_thresh)[0]).astype(np.int64)
        index_par = (np.where(angles.T[i]<par_thresh)[0]).astype(np.int64)
        #print(len(index_par), len(index_per))

        PSD_par[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_par])**2) ) * ( 2*dt)
        PSD_per[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_per])**2) ) * ( 2*dt)

    return PSD_par, PSD_per


def trace_PSD_wavelet(
        x, y, z, dt, dj,  
        mother_wave='morlet', 
        consider_coi=True,
        keep_ind = None
    ):
    """
    Method to calculate the  power spectral density using wavelet method.
    Parameters
    ----------
    x,y,z: array-like
        the components of the field to apply wavelet tranform
    dt: float
        the sampling time of the timeseries
    dj: determines how many scales are used to estimate wavelet coeff
    
        (e.g., for dj=1 -> 2**numb_scales 
    mother_wave: str
        The main waveform to transform data.
        Available waves are:
        'gaussian':
        'paul': apply lomb method to compute PSD
        'mexican_hat':
    consider_coi : boolean
        consider coi or not when calculating trace PSD
    keep_ind : array-like
        same length as x,y,z, values are boolean True/False
        the points labeled as False will be ignored upon averaging
    Returns
    -------
    db_x,db_y,db_zz: array-like
        component coeficients of th wavelet tranform
    freq : array-like
        Frequency of the corresponding psd points.
    psd : array-like
        Power Spectral Density of the signal.
    scales : array-like
        The scales at which wavelet was estimated
    coi : array-like
        The Cone of Influence (CoI)
    """
    

    if mother_wave in mother_wave_dict.keys():
        mother_morlet = mother_wave_dict[mother_wave]
    else:
        mother_morlet = wavelet.Morlet()
        
    N                                       = len(x)

    db_x, scales, freqs, coi, _, _ = wavelet.cwt(x, dt, dj, wavelet=mother_morlet)
    db_y, scales, freqs, coi, _, _ = wavelet.cwt(y, dt, dj, wavelet=mother_morlet)
    db_z, scales, freqs, coi, _, _ = wavelet.cwt(z, dt, dj, wavelet=mother_morlet)


    if consider_coi:
        # Estimate trace PSD with coefficients under COI
        try:
            PSD = np.zeros_like(freqs)
            for i1 in range(len(scales)):
                scale = scales[i1]
                ind = coi > scale
                if keep_ind is None:
                    pass
                else:
                    # ignore points labeled as False in keep_ind
                    ind = ind & keep_ind
                if np.sum(ind)/len(ind) > 0.0:
                    PSD[i1] = (
                        np.nanmean(np.abs(db_x[i1, ind])**2) + 
                        np.nanmean(np.abs(db_y[i1, ind])**2) + 
                        np.nanmean(np.abs(db_z[i1, ind])**2)
                    )*( 2*dt)
                else:
                    PSD[i1] = np.nan
        except:
            raise ValueError("failed at i1 = %d" %(i1))
    else:
        # Estimate trace powerspectral density without considering COI
        PSD = (np.nanmean(np.abs(db_x)**2, axis=1) + np.nanmean(np.abs(db_y)**2, axis=1) + np.nanmean(np.abs(db_z)**2, axis=1)   )*( 2*dt)
    
    # # Also estimate the scales to use later
    # scales = ((1/freqs)/dt)#.astype(int)
    
    return db_x, db_y, db_z, freqs, PSD, scales, coi



# -----------  Drawing ----------- #

def DrawShadedEventInTimeSeries(interval, axes, color = 'red', alpha = 0.02, lw = 2):
    """
    Draw shaded event area in time series
    Input:
        interval: dict, must have keys: start_time, end_time, spacecraft
        axes: axes to plot on
    Keyword:
        color: default red
    """

    # start time
    x = interval['start_time']
    red_vline = {
        'timestamp': interval['start_time'],
        'x_period': None,
        'lines': {}
        }
    for k, ax in axes.items():
        red_vline['lines'][k] = ax.axvline(x, color = color, ls = '--', lw = lw)
    
    interval['lines1'] = red_vline['lines']

    # end time
    x = interval['end_time']
    red_vline = {
        'timestamp': interval['end_time'],
        'x_period': None,
        'lines': {}
        }
    for k, ax in axes.items():
        red_vline['lines'][k] = ax.axvline(x, color = color, ls = '--', lw = lw)
    
    interval['lines2'] = red_vline['lines']

    if 'rects' not in interval.keys():
        interval['rects'] = {}
    for k, ax in axes.items():
        interval['rects'][k] = ax.axvspan(interval['start_time'], interval['end_time'], alpha = alpha, color = color)

    return interval



# -----------  MISC ----------- #

def PreloadDiagnostics(
        selected_interval, p_funcs, 
        credentials = None, resolution = None, import_dfmag = None,
        rescale_mag = False, check_exist = True
    ):
    """
    Preload the Diagnostics, and return the selected_interval dictionary
    keywords:
        import_dfmag        dictionary, contains dfmag_raw, dfmag, infos
        rescale_mag         rescale magnetic field data with dist_au
    """
    si = selected_interval
    sc = si['spacecraft']
    t0 = si['start_time']
    t1 = si['end_time']

    try:
        if import_dfmag is not None:
            dfmag = import_dfmag['dfmag']
            dfmag_raw = import_dfmag['dfmag_raw']
            infos = import_dfmag['infos']
        else:
            raise ValueError("Importing dfmag failed!")

    except:
        dfmag_raw, dfmag, infos = LoadHighResMagWrapper(
            sc, t0, t1,
            credentials = credentials,
            resolution = resolution
        )

    si['dfmag'] = dfmag
    si['dfmag_raw'] = dfmag_raw
    si['high_res_infos'] = infos

    res = infos['resolution']
    Br = dfmag['Bx'].interpolate()
    Bt = dfmag['By'].interpolate()
    Bn = dfmag['Bz'].interpolate()

    # rescale magnetic field data with heliocentric distance
    if rescale_mag == True:
        # acquire radial distance from si
        try:
            r = si['Dist_au']
            Br = Br * ((r/r[0])**2)
            Bt = Bt * ((r/r[0]))
            Bn = Bn * ((r/r[0]))
        except:
            raise ValueError("No r in selected_interval['Dist_au']")

    # drop na
    Br = Br.dropna().squeeze()
    Bt = Bt.dropna().squeeze()
    Bn = Bn.dropna().squeeze()

    resample_info = {
                'Fraction_missing': dfmag['Bx'].apply(np.isnan).sum()/len(dfmag['Bx'])*100,
                'resolution': res
            }
    si['resample_info'] = resample_info

    if check_exist:
        print("Keeping the diagnostics!!")

    if 'PSD' in p_funcs.keys():
        if (check_exist) & ('PSD' in si.keys()):
            print("Skipping PSD")
        else:
            freqs, PSD = TracePSD(Br.values, Bt.values, Bn.values, res)
            _, sm_freqs, sm_PSD = smoothing_function(freqs, PSD)

            si['PSD'] = {
                'freqs': freqs,
                'PSD': PSD,
                'sm_freqs': sm_freqs,
                'sm_PSD': sm_PSD,
                'diagnostics': {}
            }

    if 'struc_funcs' in p_funcs.keys():
        if (check_exist) & ('struc_funcs' in si.keys()):
            print("Skipping struc_funcs")
            pass
        else:
            maxtime = np.log10((t1-t0)/pd.Timedelta('1s')/2)
            mintime = np.log10(2*res)
            struc_funcs = MagStrucFunc(Br, Bt, Bn, (mintime, maxtime), 200)

            si['struc_funcs'] = struc_funcs
            si['struc_funcs']['diagnostics'] = {}
            si['struc_funcs']['settings'] = {
                'mintime': mintime,
                'maxtime': maxtime,
                'npts': 200
            }

    if 'wavelet_PSD' in p_funcs.keys():
        if (check_exist) & ('wavelet_PSD' in si.keys()):
            print("Skipping wavelet_PSD")
            pass
        else:
            if 'coi_thresh' in p_funcs['wavelet_PSD'].keys():
                coi_thresh = p_funcs['wavelet_PSD']['coi_thresh']
            else:
                coi_thresh = 0.8

            _,_,_,freqs_wl,PSD_wl,scales,coi = trace_PSD_wavelet(
                Br.values, Bt.values, Bn.values, res, 
                dj = 1./12
            )
            freqs_FFT, PSD_FFT = TracePSD(Br.values, Bt.values, Bn.values, res)
            _, sm_freqs_FFT, sm_PSD_FFT = smoothing_function(freqs_FFT, PSD_FFT)

            si['wavelet_PSD'] = {
                'freqs': freqs_wl,
                'PSD': PSD_wl,
                'scales': scales,
                'coi': coi,
                'freqs_FFT': freqs_FFT,
                'PSD_FFT': PSD_FFT,
                'sm_freqs_FFT': sm_freqs_FFT,
                'sm_PSD_FFT': sm_PSD_FFT,
                'diagnostics':{},
                'settings':{
                    'dj': 1./12
                }
            }



def UpdatePSDDict(path, credentials = None, loadSPEDASsettings = None):
    """
    Update the PSD dictionary with high resolution magnetic field time series and spectrum, and the full time series, and fix some bugs....
    """
    # for loading time series
    from TimeSeriesViewer import TimeSeriesViewer

    # default keys of the dictionary
    default_keys = ['spacecraft', 'start_time', 'end_time', 'TimeSeries', 'PSD']

    # load the PSD from path
    d = pd.read_pickle(Path(path).joinpath("bpf/bpf.pkl"))

    # check keys
    for k in default_keys:
        if k in d.keys():
            pass
        else:
            raise ValueError("The dictionary is not complete... Need at least the following keys:" + "%s, " * len(default_keys) %(tuple(default_keys)))

    # load spacecraft and start/end time info
    sc = d['spacecraft']
    tstart = d['start_time']
    tend = d['end_time']

    # load the time series with TimeSeriesViewer Module
    start_time_0 = tstart
    end_time_0 = tend
    tsv = TimeSeriesViewer(
        sc = sc,
        start_time_0 = start_time_0, 
        end_time_0 = end_time_0,
        credentials = credentials,
        loadSPEDASsettings = loadSPEDASsettings
    )

    # generate the time series
    tsv.InitFigure(tstart, tend, no_plot=True)

    # store the new time series
    d['TimeSeries'] = tsv.dfts

    # Load magnetic field
    t0 = tstart.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = tend.strftime("%Y-%m-%d/%H:%M:%S")
    if sc == 0:
        # Magnetic field
        try:
            names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_rtn', level='l2', time_clip=True)
            data = get_data(names[0])
            dfmag1 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag1.columns = ['Br','Bt','Bn']

            names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_sc', level='l2', time_clip=True)
            data = get_data(names[0])
            dfmag2 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag2.columns = ['Bx','By','Bz']
        except:
            print("No MAG data is presented in the public repository!")
            print("Trying unpublished data... please provide credentials...")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            names = pyspedas.psp.fields(trange=[t0,t1], 
                datatype='mag_RTN', level='l2', time_clip=True,
                username=username, password=password
            )
            data = get_data(names[0])
            dfmag1 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag1.columns = ['Br','Bt','Bn']

            names = pyspedas.psp.fields(trange=[t0,t1], 
                datatype='mag_SC', level='l2', time_clip=True,
                username=username, password=password
            )
            data = get_data(names[0])
            dfmag2 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag2.columns = ['Bx','By','Bz']

        dfmag = dfmag1.join(dfmag2)
        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)

    elif sc == 1:
        names = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-normal', level='l2', time_clip=True)
        data = get_data(names[0])
        dfmag1 = pd.DataFrame(
            index = data[0],
            data = data[1]
        )
        dfmag1.columns = ['Br','Bt','Bn']

        names = pyspedas.solo.mag(trange=[t0,t1], datatype='srf-normal', level='l2', time_clip=True)
        data = get_data(names[0])
        dfmag2 = pd.DataFrame(
            index = data[0],
            data = data[1]
        )
        dfmag2.columns = ['Bx','By','Bz']
        
        dfmag = dfmag1.join(dfmag2)
        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)
    else:
        raise ValueError("sc=%d not supported!" %(sc))

    # determine resample frequency
    val, counts = np.unique(np.diff(dfmag.index), return_counts=True)

    # gaps are not counted as real cadence hence remove gap jumps
    # val are in timedelta64[ns], cadance > 5s are regarded as gaps
    val = val[val.astype(float) <= 5e9]

    # find the maximum of the cadence, and resample to the corresponding freq
    if (np.max(val).astype(float)/1e6) < 10:
        period = 1e-2
        dfmag_r = dfmag.resample('10ms').mean()
    elif (np.max(val).astype(float)/1e6) < 100:
        # 100 ms
        period = 1e-1
        dfmag_r = dfmag.resample('100ms').mean()
    elif (np.max(val).astype(float)/1e6) < 1000:
        # 1000 ms
        period = 1e0
        dfmag_r = dfmag.resample('1000ms').mean()
    else:
        period = 5
        dfmag_r = dfmag.resample('5s').mean()

    Bx = dfmag_r['Bx'].interpolate().dropna().values
    By = dfmag_r['By'].interpolate().dropna().values
    Bz = dfmag_r['Bz'].interpolate().dropna().values
    freq, B_pow = TracePSD(Bx, By, Bz, period)
    _, sm_freqs, sm_PSD = smoothing_function(freq, B_pow)

    resample_info = {
        'Fraction_missing': dfmag_r['Br'].apply(np.isnan).sum()/len(dfmag_r['Br'])*100,
        'resolution': period*1000,
        'resolution_dim': 'ms'
    }

    d['PSD']['freqs'] = freq
    d['PSD']['PSD'] = B_pow
    d['PSD']['sm_freqs'] = sm_freqs
    d['PSD']['sm_PSD'] = sm_PSD
    d['PSD']['resample_info'] = resample_info

    return d


def FindDiagnostics(
    sc, start_time, end_time, 
    settings = None, credentials = None, verbose=False
    ):
    """
    Return the diagnostics of a given interval.
    Diagnostics will be return in a dictionary:

    Input:
    sc                          spacecraft
    start_time                  start time in time stamp
    end_time                    end time in time stamp
    
    Keywords:
    settings                    dictionary of settings
    credentials                 credentials for timeseries viewer

    Return:
    {
        'sc':                   spacecraft
        'start_time':           start time in timestamp
        'end_time':             end time in timestamp
        'timeseries':           dataframe from tsv
        'diagnostics':          dictionary
    }
    list of diagnostics:
    ['Dist_au','sc_vel','vsw','di','rho_ci','beta','np','sigma_c','sigma_r','vth','valfven']
    besides 'Dist_au' and 'sc_vel', all the remaining diagnostics would be accompanied by their std
    """

    from TimeSeriesViewer import TimeSeriesViewer

    default_settings = {
        'loadSPEDASsettings': {
            'interpolate_rolling': True,
            'verbose': False,
            'particle_mode': 'empirical'
        },
        'rolling_rate': '1H',
        'resample_rate': '5s',
        'resolution': '5s',
        'window_size': '12H'
    }
    if settings is None:
        settings = {}
        for k, v in default_settings.items():
            settings[k] = v
    else:
        for k in default_settings.keys():
            if k in settings.keys():
                pass
            else:
                raise ValueError("Warning: necessary key: %s not in settings!" %(k))

    # window size for time series viewer object
    default_window_size = pd.Timedelta('24H')
    if 'window_size' not in settings.keys():
        ws = default_window_size
    else:
        ws = settings['window_size']
    
    # determine the start and end time
    if 5*(end_time-start_time) < pd.Timedelta(ws):
        # if 5*dt < 24H, use 24 Hr window
        dt = end_time - start_time
        mid_time = start_time + dt/2
        start_time_0 = mid_time - pd.Timedelta('12H')
        end_time_0 = mid_time + pd.Timedelta('12H')
    else:
        # use 5*dt window
        dt = end_time-start_time
        start_time_0 = start_time - 2*dt
        end_time_0 = end_time + 2*dt

    # create time series object
    try:
        tsv = TimeSeriesViewer(
            sc=sc,start_time_0=start_time_0, end_time_0=end_time_0,
            credentials = credentials,
            loadSPEDASsettings = settings['loadSPEDASsettings'],
            resample_rate = settings['resample_rate'],
            rolling_rate = settings['rolling_rate'],
            resolution = settings['resolution'],
            verbose=verbose
        )
    except:
        raise ValueError("Time Series Viewer Initialization Failed! Lack of data...?")

    # generate diagnostics time series
    t0 = start_time
    t1 = end_time
    tsv.InitFigure(t0, t1, no_plot = True)

    # return the timeseries
    dfts = tsv.dfts

    # calculate diagnostics
    keys = ['vsw','brangle','vbangle','Dist_au','tadv','sigma_c','sigma_r','np','di','rho_ci','Vth','beta','B','Vr','Vt','Vn','valfven']

    diagnostics = {}
    for k in keys:
        diagnostics[k] = np.nanmean(dfts[k])
        diagnostics[k+'_std'] = np.nanstd(dfts[k])

    resample_info = {}
    mag_missing = dfts['B'].apply(np.isnan).sum()/len(dfts)
    par_missing = dfts['np'].apply(np.isnan).sum()/len(dfts)
    resample_info['mag_missing'] = mag_missing
    resample_info['par_missing'] = par_missing

    d = {
        'sc': sc,
        'start_time': start_time,
        'end_time': end_time,
        'start_time_0': start_time_0,
        'end_time_0': end_time_0,
        'settings': settings,
        'timeseries': dfts,
        'diagnostics': diagnostics,
        'resample_info': resample_info
    }


    return d



# ------- obsolete ------- #
# def FindIntervalInfo(sc, start_time, end_time, verbose = False, spdf = False, local = False, tsv = None, keys = None):

#     if spdf:
#         spdf_data = LoadTimeSeriesSPDF(sc = sc, 
#             start_time=start_time.to_pydatetime(), 
#             end_time = end_time.to_pydatetime(), 
#             verbose = verbose, keys = keys)

#         d = {
#             'sc': sc,
#             'start_time': start_time,
#             'end_time': end_time,
#             'spdf_data': spdf_data,
#             'Diagnostics': None
#         }
#         return d

#     if local:
#         try:
#             if sc == tsv.sc:
#                 pass
#             else:
#                 raise ValueError("sc = %d must equal tsv.sc = %d" %(sc, tsv.sc))
#         except:
#             raise ValueError("tsv must be a TimeSeriesViewer Object!!!")

#         dfts = tsv.ExportTimeSeries(start_time, end_time, verbose = verbose)

#         d = {
#             'sc': sc,
#             'start_time': start_time,
#             'end_time': end_time,
#             'local_data': {
#                 'dfts_raw': tsv.dfts_raw
#             },
#             'dfts': dfts,
#             'Diagnostics': None
#         }
#         return d


# @jit(nopython=True, parallel=True)      
# def smoothing_function_obsolete(x,y, window=2, pad = 1):
#     xoutmid = np.zeros(len(x))*np.nan
#     xoutmean = np.zeros(len(x))*np.nan
#     yout = np.zeros(len(x))*np.nan
#     for i in prange(len(x)):
#         x0 = x[i]
#         xf = window*x[i]
        
#         if xf < np.max(x):
#             e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
#             if e<len(x):
#                 yout[i] = np.nanmean(y[i:e])
#                 xoutmid[i] = x[i] + np.log10(0.5) * (x[i] - x[e])
#                 xoutmean[i] = np.nanmean(x[i:e])
#     return xoutmid, xoutmean, yout 
