# %matplotlib tk
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from packaging import version
from scipy.interpolate import interp1d

import numpy as np
from scipy import stats
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
import warnings
# from hampel import hampel

from datetime import datetime

from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon

# SPDF API
from cdasws import CdasWs
cdas = CdasWs()

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

from TSUtilities import SolarWindCorrelationLength, DrawShadedEventInTimeSeries, smoothing_function, TracePSD, PreloadDiagnostics, trace_PSD_wavelet, hampel
from BreakPointFinderGeneral import BreakPointFinder as BPFG
from LoadData import LoadTimeSeriesWrapper, LoadHighResMagWrapper
from WaveletRainbow import WaveletRainbow
from scipy.stats import shapiro, kstest
# from BreakPointFinderLite import BreakPointFinder
# from StructureFunctions import MagStrucFunc
# from TurbPy import trace_PSD_wavelet

class TimeSeriesViewer:
    """ 
    View Time Series and Select Intervals 
    start_time_0/end_time_0 are pd.timestamp
    """

    def __init__(
        self,  
        load_spdf = False, 
        start_time_0 = None, 
        end_time_0 = None, 
        sc = 0, 
        preload = True, 
        verbose = True,
        rolling_rate = '1H',
        resample_rate = '5min',
        resolution = 5,
        credentials = None,
        p_funcs = {'PSD':True},
        LTSWsettings = {},
        layout = None,
        high_res_resolution = None,
        use_hampel_raw = True,
        hampel_settings = None,
        no_high_res_mag = False,
    ):
        """ Initialize the class """

        # check compatibility
        if version.parse(matplotlib.__version__) >= version.parse('3.7.0'):
            pass
        else:
            raise ValueError("Matplotlib version: %s, required version >= %s" %(matplotlib.__version__, '3.7.0')) 

        print("Initializing...")
        print("Current Directory: %s" %(os.getcwd()))
        
        # session info
        self.start_time_0 = start_time_0
        self.end_time_0 = end_time_0
        self.sc = sc

        # dataframe:
        self.dfts_raw = None
        self.dfts_misc = None

        # misc
        self.load_spdf = load_spdf
        self.resample_rate_key = 1
        self.resample_rate_list = ['1min', '5min','10min','20min','30min','60min']
        self.length_list = ['12H','24H','2d','5d','10d','20d','30d','60d','120d','365d']
        self.p_funcs_list = ['PSD','struc_funcs','wavelet_PSD']
        self.length_key = 2
        self.red_vlines = []
        self.selected_intervals = []
        self.green_vlines = []
        self.window_size_histories = []
        self.verbose = verbose
        self.rolling_rate = rolling_rate
        self.resample_rate = resample_rate
        self.resolution = resolution # for function p, resolution in seconds
        self.mag_option = {'norm':0, 'sc':0}
        self.spc_only = True
        self.calc_smoothed_spec = True
        self.useBPF = True
        self.credentials = credentials
        self.LTSWsettings = LTSWsettings
        self.p_funcs = p_funcs
        self.layout = layout
        self.import_timeseries=None
        self.use_hampel = False
        self.use_hampel_raw = use_hampel_raw
        self.hampel_settings = hampel_settings
        self.figsize = None
        self.no_high_res_mag = no_high_res_mag

        self.high_res_resolution = high_res_resolution
        self.skipPreLoadDiagnostics = False

        # Preload Time Series
        if preload:
            if verbose: print("Preloading raw dataframe... This may take some time...")
            self.PreLoadSCDataFrame()
            if verbose: print("Done.")

        # collect garbage
        collect()

        print("Done.")


    def PreLoadSCDataFrame(self): 
        """ 
        Preload Spacecraft dataframe 
        """

        # Load Spacecraft data with wrapper
        dfts_raw0, misc = LoadTimeSeriesWrapper(self.sc, self.start_time_0, self.end_time_0, 
            credentials = self.credentials, 
            settings = self.LTSWsettings
        )

        if self.sc == 0:
            # psp has 4 per cyc data, enough for our use
            dfmag_raw = misc['dfmag'][['Br','Bt','Bn']]
            dfmag_raw.columns = ['Bx','By','Bz']
            dfmag_raw['Btot'] = (dfmag_raw[['Bx','By','Bz']]**2).sum(axis=1).apply(np.sqrt)
            self.dfmag_raw_high_res = dfmag_raw
            dfmag1 = dfmag_raw.resample("500ms").mean()
            self.dfmag_high_res = dfmag1
            self.dfmag_infos =  {
                'resolution': 0.5,
                'fraction_missing': np.isnan(dfmag1['Bx']).sum()/len(dfmag1)
            }

        else:
            if self.no_high_res_mag:
                dfmag_raw = dfts_raw0[['Br','Bt','Bn']]
                dfmag_raw['Btot'] = (dfmag_raw[['Br','Bt','Bn']]**2).sum(axis=1).apply(np.sqrt)
                dfmag = dfmag_raw.copy()
                infos = {'resolution': 60}
            else:
                # Load High Res Magnetic field
                dfmag_raw, dfmag, infos = LoadHighResMagWrapper(
                    self.sc, self.start_time_0, self.end_time_0, credentials = self.credentials
                )
                self.dfmag_raw_high_res = dfmag_raw
                self.dfmag_high_res = dfmag
                self.dfmag_infos = infos

        # calculate the averged field
        keys = ['Br','Bt','Bn','Vr','Vt','Vn','Bx','By','Bz','Vx','Vy','Vz']

        for k in keys:
            if k in dfts_raw0.columns:
                dfts_raw0[k+'0'] = dfts_raw0[k].rolling(self.rolling_rate).mean()
            else:
                print("%s not in columns...!" %(k))

        # for parmode = 'keep_all', conduct additional rolling...
        if (self.sc == 0):
            if (misc['parmode'] == 'keep_all'):
                print("Parmode: keep_all, additional rolling...")
                # keys = [mom+'_'+ins for mom in ['Vr','Vt','Vn','Vx','Vy','Vz'] for ins in ['spc','span']]
                for mom in ['Vr','Vt','Vn','Vx','Vy','Vz']:
                    for ins in ['spc','span']:
                        k = mom+'_'+ins
                        if k in dfts_raw0.columns:
                            print("Rolling: %s" %(k))
                            dfts_raw0[mom+"0"+"_"+ins] = dfts_raw0[k].rolling(self.rolling_rate).mean()
                        else:
                            print("%s not in columns...!" %(k))

        # apply hampel filter
        if self.use_hampel_raw == True:

            self.hampel_results = {}

            print("Applying Hampel filter...")
            if self.hampel_settings is not None:
                ws_hampel = self.hampel_settings['window_size']
                n_hampel = self.hampel_settings['n']
                ins_list = self.hampel_settings['ins_list']
            else:
                ws_hampel = 100
                n_hampel = 3
                ins_list = ['span','spc']

            print("Window size: %s, n: %s" %(ws_hampel, n_hampel))

            for kraw in ['Vr','Vt','Vn','np','Vth']:
                for ins in ins_list:
                    k = kraw+'_'+ins
                    try:
                        outliers_indices = hampel(dfts_raw0[k], window_size = ws_hampel, n = n_hampel)
                        # print(outliers_indices)
                        dfts_raw0.loc[dfts_raw0.index[outliers_indices], k] = np.nan
                        self.hampel_results[k] = {
                            'window_size': ws_hampel,
                            'n': n_hampel,
                            'outliers_indices': outliers_indices,
                            'outliers_values': dfts_raw0[k][outliers_indices],
                            'outliers_index': dfts_raw0.index[outliers_indices],
                            'outliers_count': len(outliers_indices),
                            'total_count': len(dfts_raw0[k]),
                            'outliers_percent': float(len(outliers_indices))/len(dfts_raw0[k]),
                        }
                        print("Filtering: %s, outliers ratio: %.4f ppm"%(k, 1e6*float(len(outliers_indices))/len(dfts_raw0[k])))
                    except:
                        # raise ValueError("FUCK")
                        print("key: %s does not exist!" %(k))

        # setting value
        self.dfts_raw0 = dfts_raw0
        self.dfts_misc = misc



    def PrepareTimeSeries(
        self, start_time, end_time, 
        update_resample_rate_only = False
        ):
        """ One function to prepare time series """
        if update_resample_rate_only:
            pass
        else:
            if self.verbose: print("Finding Corresponding Time Series...")
            self.FindTimeSeries(start_time, end_time)

        if self.verbose: print("Processing the Time Series...")
        self.ProcessTimeSeries()


    def FindTimeSeries(
        self, start_time, end_time
        ):
        """ 
        Find time series in dataframe and create a dfts dataframe 
        start_time and end_time should be either datetime64[ns] or timestamp
        sc: 0-PSP, 1-SolO, 2-Helios1, 3-Helios2

        par_settings: Dictionary
            allowed settings:
                'density': 'QTN'/'SPC'/'SPAN'
                'moments': 'SPC'/'SPAN'/'empirical'
            Note: 'empirical' use 'SPC' before 2021-07-01, and 'SPAN' afterwards

        """

        # find the time series between start and end time
        ind = (self.dfts_raw0.index > start_time) & (self.dfts_raw0.index < end_time) 

        # store the time series dataframe
        self.dfts_raw = self.dfts_raw0[ind].copy()

        # if parmode = 'keep_all', now choose particle data, default is empirical
        par_settings = self.par_settings
        if (self.sc == 0):
            if (self.dfts_misc['parmode'] == 'keep_all'):
                if self.verbose:
                    print("sc = %d, Parmode: keep_all" %(self.sc))
                    print("Current Partical settings:")
                    for k, v in par_settings.items():
                        print("%s: %s" %(k, v))

                    if 'density' in par_settings.keys():
                        if par_settings['density'] == 'QTN':
                            self.dfts_raw['np'] = self.dfts_raw['np_qtn']
                        elif par_settings['density'] == 'SPC':
                            self.dfts_raw['np'] = self.dfts_raw['np_spc']
                        elif par_settings['density'] == 'SPAN':
                            self.dfts_raw['np'] = self.dfts_raw['np_span']
                        else:
                            warnings.warn("density = %s not supported, use default QTN!" %(par_settings['density']))
                            self.dfts_raw['np'] = self.dfts_raw['np_qtn']
                    
                    mom_keys = ['Vx','Vy','Vz','Vr','Vt','Vn','Vth', 'Vx0','Vy0','Vz0','Vr0','Vt0','Vn0']
                    if 'moments' in par_settings.keys():
                        if par_settings['moments'] == 'SPC':
                            for k in mom_keys: self.dfts_raw[k] = self.dfts_raw[k+'_spc']
                        elif par_settings['moments'] == 'SPAN':
                            for k in mom_keys: self.dfts_raw[k] = self.dfts_raw[k+'_span']
                        elif par_settings['moments'] == 'empirical':
                            if end_time < pd.Timestamp("2021-07-01"):
                                for k in mom_keys: self.dfts_raw[k] = self.dfts_raw[k+'_spc']
                            else:
                                for k in mom_keys: self.dfts_raw[k] = self.dfts_raw[k+'_span']


        # collect garbage
        collect()



    def CheckTimeBoundary(self):
        """ check the boundary of time series """
        if self.start_time < self.start_time_0:
            self.start_time = self.start_time_0
        
        if self.end_time > self.end_time_0:
            self.end_time = self.end_time_0

        if self.end_time_0 - self.end_time < pd.Timedelta('1H'):
            self.start_time = np.max([self.start_time,self.end_time]) - pd.Timedelta(self.length_list[self.length_key])

        if self.start_time - self.start_time_0 < pd.Timedelta('1H'):
            self.end_time = np.min([self.start_time,self.end_time]) + pd.Timedelta(self.length_list[self.length_key])


    #-------- Create the Plot --------#


    def ProcessTimeSeries(self):
        """ Process time series to produce desired data product """

        dfts = self.dfts_raw.copy()
        resample_rate = self.resample_rate
        rolling_rate = self.rolling_rate
        verbose = self.verbose

        """resample the time series"""
        dfts = dfts.resample(resample_rate).mean()

        # remove outliers using hampel filter
        if self.use_hampel:
            for k in ['Vr','Vt','Vn','np','Vth']:
                print("Filtering... "+k)
                if self.hampel_settings is not None:
                    ws_hampel = self.hampel_settings['window_size']
                    n_hampel = self.hampel_settings['n']
                else:
                    ws_hampel = 100
                    n_hampel = 3
                
                outliers_indices = hampel(dfts[k], window_size = ws_hampel, n = n_hampel)
                dfts.loc[dfts.index[outliers_indices], k] = np.nan


        """Modulus of vectors"""
        dfts['B_RTN'] = (dfts['Br']**2+dfts['Bt']**2+dfts['Bn']**2).apply(np.sqrt)
        dfts['V_RTN'] = (dfts['Vr']**2+dfts['Vt']**2+dfts['Vn']**2).apply(np.sqrt)
        try:
            dfts['B_SC'] = (dfts['Bx']**2+dfts['By']**2+dfts['Bz']**2).apply(np.sqrt)
        except:    
            print("SC MAG data not exist! Use RTN instead!")
            dfts['B_SC'] = (dfts['Br']**2+dfts['Bt']**2+dfts['Bn']**2).apply(np.sqrt)
        try:
            dfts['V_SC'] = (dfts['Vx']**2+dfts['Vy']**2+dfts['Vz']**2).apply(np.sqrt)
        except:
            print("SC SW data not exist! Use RTN instead!")
            dfts['V_SC'] = (dfts['Vr']**2+dfts['Vt']**2+dfts['Vn']**2).apply(np.sqrt)
        
        dfts['B'] = dfts['B_SC']
        dfts['V'] = dfts['V_SC']

        """br angle"""
        dfts['brangle'] = (dfts['Br']/dfts['B_RTN']).apply(np.arccos) * 180 / np.pi

        """Diagnostics"""

        Bmod = (dfts['Br']**2 + dfts['Bt']**2 + dfts['Bn']**2).apply('sqrt')

        dfts['Br0'] = dfts['Br'].rolling(rolling_rate).mean()
        dfts['Bt0'] = dfts['Bt'].rolling(rolling_rate).mean()
        dfts['Bn0'] = dfts['Bn'].rolling(rolling_rate).mean()
        dfts['Vr0'] = dfts['Vr'].rolling(rolling_rate).mean()
        dfts['Vt0'] = dfts['Vt'].rolling(rolling_rate).mean()
        dfts['Vn0'] = dfts['Vn'].rolling(rolling_rate).mean()

        Bmod0 = (dfts['Br0']**2 + dfts['Bt0']**2 + dfts['Bn0']**2).apply('sqrt')

        nproton = dfts['np'].interpolate(method='linear')

        Va_r = 1e-15* dfts['Br']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_t = 1e-15* dfts['Bt']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_n = 1e-15* dfts['Bn']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]

        Va_r0 = 1e-15* dfts['Br0']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_t0 = 1e-15* dfts['Bt0']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_n0 = 1e-15* dfts['Bn0']/np.sqrt(mu0*nproton*m_p)   ### Multuply by 1e-15 to get units of [Km/s]

        vr = dfts['Vr'].interpolate(); vt = dfts['Vt'].interpolate(); vn = dfts['Vn'].interpolate(); 
        vr0 = dfts['Vr0']; vt0 = dfts['Vt0']; vn0 = dfts['Vn0']; 

        # # Estimate fluctuations of fields #
        # va_r = Va_r - np.nanmean(Va_r);   v_r = vr - np.nanmean(vr)
        # va_t = Va_t - np.nanmean(Va_t);   v_t = vt - np.nanmean(vt)
        # va_n = Va_n - np.nanmean(Va_n);   v_n = vn - np.nanmean(vn)

        # Use moving mean to estimate fluctuation of fields
        va_r = Va_r - Va_r0   
        va_t = Va_t - Va_t0   
        va_n = Va_n - Va_n0   
        v_r = vr - vr0
        v_t = vt - vt0
        v_n = vn - vn0

        Z_plus_squared  = (v_r + va_r)**2 +  (v_t + va_t)**2 + ( v_n + va_n)**2
        Z_minus_squared = (v_r - va_r)**2 +  (v_t - va_t)**2 + ( v_n - va_n)**2
        Z_amplitude     = np.sqrt( (Z_plus_squared + Z_minus_squared)/2 )    

        # cross helicity
        sigma_c         =  (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared)
        sigma_c[np.abs(sigma_c) > 1e5] = np.nan
        dfts['sigma_c'] = sigma_c

        # alfvenicity
        dBvecmod = np.sqrt(dfts['Br'].diff()**2 + dfts['Bt'].diff()**2 + dfts['Bn'].diff()**2)
        dBmod = np.sqrt(dfts['Br']**2+dfts['Bt']**2+dfts['Bn']**2).diff()
        alfvenicity = dBmod/dBvecmod
        dfts['alfvenicity'] = alfvenicity

        # Residual energy
        Ek = v_r**2 + v_t**2 + v_n**2
        Eb = va_r**2 + va_t**2 + va_n**2
        sigma_r = (Ek-Eb)/(Ek+Eb)
        sigma_r[np.abs(sigma_r) > 1e5] = np.nan
        dfts['sigma_r'] = sigma_r

        # V B angle
        vbang = np.arccos(
            (Va_r * vr + Va_t * vt + Va_n * vn)
            /
            np.sqrt(
                (Va_r**2+Va_t**2+Va_n**2)
                *
                (vr**2+vt**2+vn**2)
            )
        )
        vbang = vbang/np.pi*180
        # vbang[vbang>90] = 180 - vbang[vbang>90]
        dfts['vbangle'] = vbang

        # ion inertial length
        di = 228/np.sqrt(dfts['np']) #in [km]
        di[di < 0] = np.nan
        di[np.log10(di) < -3] = np.nan
        dfts['di'] = di

        # ion gyro radius
        rho_ci = 10.43968491 * dfts['Vth']/Bmod #in [km]
        rho_ci[rho_ci < 0] = np.nan
        rho_ci[np.log10(rho_ci) < -3] = np.nan
        dfts['rho_ci'] = rho_ci

        # Plasma Beta
        km2m        = 1e3
        nT2T        = 1e-9
        cm2m        = 1e-2
        B_mag       = Bmod * nT2T                           # |B| units:      [T]
        temp        = 1./2 * m_p * (dfts['Vth']*km2m)**2      # in [J] = [kg] * [m]^2 * [s]^-2
        dens        = dfts['np']/(cm2m**3)                    # number density: [m^-3] 
        beta        = (dens*temp)/((B_mag**2)/(2*mu_0))         # plasma beta
        beta[beta < 0] = np.nan
        beta[np.abs(np.log10(beta))>4] = np.nan # delete some weird data
        dfts['beta'] = beta  

        # SW speed
        # vsw = np.sqrt(vr**2+vt**2+vn**2)
        vsw = vr
        vsw[vsw < 0] = np.nan
        vsw[np.abs(vsw) > 1e5] = np.nan
        dfts['vsw'] = vsw

        # deal with weird values
        vth = dfts['Vth'].to_numpy()
        vth[vth < 0] = np.nan
        dfts['Vth'] = vth

        # # distance
        # try:
        #     dist = dfts['Dist_au'].to_numpy()
        #     dist_spdf = dfts['RAD_AU'].to_numpy()
        #     dist[np.isnan(dist)] = dist_spdf[np.isnan(dist)]
        #     dfts['Dist_au'] = dist
        # except:
        #     pass

        # advection time [Hr]
        # 1Rsun = 0.00465047AU
        tadv = ((dfts['Dist_au']-0.00465047)/vr).to_numpy() * au_to_km /3600
        tadv[tadv < 0] = np.nan
        dfts['tadv'] = tadv
        # try:
        #     tadv_spdf = (dfts['RAD_AU']/dfts['vsw']).to_numpy() * au_to_km /3600
        #     tadv_spdf[tadv_spdf < 0] = np.nan
        #     dfts['tadv_spdf'] = tadv_spdf
        #     tadv[np.isnan(tadv)] = tadv_spdf[np.isnan(tadv)]
        #     dfts['tadv'] = tadv
        # except:
        #     pass

        # alfven speed [km/s]
        valfven = dfts['B']*nT2T/np.sqrt(dfts['np']*1e6*m_p*mu0)/1e3
        dfts['valfven'] = valfven

        # alfven mach number
        dfts['malfven'] = vr/dfts['valfven']

        # na/np ratio
        # might not be available for some spacecrafts
        try:
            nanp_ratio = dfts['na']/dfts['np']
            ind =  nanp_ratio > 0.2
            nanp_ratio[ind] = np.nan
            dfts['nanp_ratio'] = nanp_ratio
        except:
            print("No na/np ratio!! All NaNs")
            nanp_ratio = dfts['np'] * np.nan
            dfts['nanp_ratio'] = nanp_ratio

        # update dfts
        self.dfts = dfts

    
    def AxesInit(self):
        """ Initialize Axes """
        if self.figsize is None:
            figsize = [20,10]
        else:
            figsize = self.figsize
        
        if self.import_timeseries is None:
            fig, axes = plt.subplots(6,1, figsize = figsize, layout=self.layout)
        else:
            fig, axes = plt.subplots(7,1, figsize = figsize, layout=self.layout)
        
        self.fig = fig
        self.axeslist = axes
        self.axes = {}
        self.lines = {}

        # assign the axes
        if self.import_timeseries is None:
            self.axes['mag'] = axes[0]              # magnetic field
            self.axes['vsw'] = axes[1]              # just vsw, no log
            self.axes['norm'] = axes[2]             # normalized quantities: sigma_c, sigma_r
            self.axes['density'] = axes[3]          # density and di and rho_ci
            self.axes['ang'] = axes[4]              # angles
            self.axes['rau'] = axes[5]              # rau
        else:
            key = self.import_timeseries['key']
            self.axes['mag'] = axes[0]              # magnetic field
            self.axes[key] = axes[1]                # import time series
            self.axes['vsw'] = axes[2]              # just vsw, no log
            self.axes['norm'] = axes[3]             # normalized quantities: sigma_c, sigma_r
            self.axes['density'] = axes[4]          # density and di and rho_ci
            self.axes['ang'] = axes[5]              # angles
            self.axes['rau'] = axes[6]              # rau

        # self.axes['spec'] =  axes[3]            # spectrum
        # self.axes['alfvenicity'] = axes[5]      # delta |B| / |delta \vec B|
        # self.axes['speed'] = axes[1],           # Vth and Vsw
        # self.axes['scale'] =  axes[3],           # di and rho_ci


    def InitFigure(self, start_time, end_time, 
        no_plot = False, 
        clean_intervals = False, 
        auto_connect=False,
        par_settings = {
            'density': 'QTN',
            'moments': 'empirical'
        },
        dfts = None,
        # mag_res = None,
        import_timeseries = None
        ):
        """ 
        Initialize figure 
        Keywords:
            no_plot: Boolean
                set to True to create dfts only
            clean_interval: Boolean
                set to True to clean self.selected_intervals
            auto_connect: Boolean
                set to True to connect to matplotlib object automatically, otherwise self.connect() manually
            par_settings: Dictionary
                defaults values is set to density=QTN, moments=empirical, will be saved as self.par_settings
                only useable for sc==0 & parmode == 'keep_all'
                allowed settings:
                    'density': 'QTN'/'SPC'/'SPAN'
                    'moments': 'SPC'/'SPAN'/'empirical'
                Note: 'empirical' use 'SPC' before 2021-07-01, and 'SPAN' afterwards
            dfts: 
                import dfts instead of creating one, will be deprecate soon...
            import_timeseries:
                {'key':str, 'df':pd.DataFrame, 'labels':list of str, columns: list of keys, styles: list of str}
                optional keys: {'ylim':np.array, 'yscale':'log'}
        """

        # default values
        self.start_time = start_time
        self.end_time = end_time

        # get settings
        verbose = self.verbose

        # set par_settings
        self.par_settings = par_settings

        # mag resolution
        # self.mag_res = mag_res

        # Prepare Time Series
        if dfts is None:
            if verbose: print("Preparing Time Series....")
            self.PrepareTimeSeries(
                start_time, end_time
                )
            if verbose: print("Done.")
        # import dfts
        else:
            if verbose: print("Importing dfts...")
            self.dfts = dfts

        # just create dfts
        if no_plot:
            return

        # Import Timeseries
        if import_timeseries is not None:
            self.import_timeseries = import_timeseries

        # Initialize Axes
        self.AxesInit()

        # Plot Time Series
        self.PlotTimeSeries()

        # Create Floating text for crosshair
        self.text = self.axes['mag'].text(0.72, 0.9, '', transform=self.axes['mag'].transAxes, fontsize = 'xx-large')

        # self.horizontal_lines = {}
        self.vertical_lines = {}

        # clean info
        self.green_vlines = []
        if clean_intervals:
            self.selected_intervals = []
        else:
            # plot the intervals
            for interval in self.selected_intervals:
                _ = DrawShadedEventInTimeSeries(interval, self.axes, color = 'red')

        # Show cross hair
        for k, ax in self.axes.items():
            # self.horizontal_lines[k] = ax.axhline(color='k', lw=0.8, ls='--')
            self.vertical_lines[k] = ax.axvline(color='k', lw=0.8, ls='--')

        # redraw
        self.fig.canvas.draw()

        # connect to mpl
        if auto_connect:
            self.connect()


    def ExportSelectedIntervals(self):
        """ 
        Export selected Intervals 
        Return:
            intervals   :   list of dictionaries
            keys        :   list of keys for dict
        """
        intervals = []
        keys = ['spacecraft', 'start_time', 'end_time', 'TimeSeries','PSD']
        print("Current Keys:")
        for k in keys: print(k)
        try:
            for interval in self.selected_intervals:
                temp = {}
                for k in keys:
                    temp[k] = interval[k]
                intervals.append(temp)
        except:
            print("Not enough intervals... len(self.selected_intervals) = %d" %(len(self.selected_intervals)))

        return intervals, keys


    def ImportSelectedIntervals(self, intervals):
        """ 
        Import selected Intervals 
        Return:
            intervals   :   list of dictionaries
                should have at least the following keys:
                spacecraft, start_time, end_time,
                preferably with QualityFlag key
            keys        :   list of keys for dict
        """

        # print red vlines
        for i1 in range(len(intervals)):
            interval = intervals[i1]

            if "QualityFlag" in interval.keys():
                if interval['QualityFlag'] == 0:
                    color = 'yellow'
                    interval = DrawShadedEventInTimeSeries(interval, self.axes, color = color)
                elif interval['QualityFlag'] == 1:
                    color = 'red'
                    interval = DrawShadedEventInTimeSeries(interval, self.axes, color = color)
                elif interval['QualityFlag'] == 4:
                    color = 'purple'
                    interval = DrawShadedEventInTimeSeries(interval, self.axes, color = color)
                else:
                    pass
            else:
                interval = DrawShadedEventInTimeSeries(interval, self.axes, color = 'red')

            if 'TimeSeries' not in interval.keys():
                interval['TimeSeries'] = None
            
            if 'PSD' not in interval.keys():
                interval['PSD'] = None

            # force the start_time and end_time to be pd.Timestamp
            interval['start_time'] = pd.Timestamp(interval['start_time'])
            interval['end_time'] = pd.Timestamp(interval['end_time'])

            self.selected_intervals.append(interval)

        return self.selected_intervals


    def UpdateFigure(
        self, start_time, end_time, 
        resample_rate = None, update_resample_rate_only = False
        ):
        """ Update Figure """

        self.start_time = start_time
        self.end_time = end_time

        if resample_rate is None:
            resample_rate = self.resample_rate
        else:
            self.resample_rate = resample_rate

        # Prepare Time Series
        if self.verbose: print("Preparing Time Series....")
        self.PrepareTimeSeries(
            start_time, end_time, 
            update_resample_rate_only = update_resample_rate_only
            )
        if self.verbose: print("Done.")

        # Update Time Series
        self.PlotTimeSeries(update=True)

        # redraw
        self.fig.canvas.draw()


    def PlotTimeSeries(self, verbose = False, update=False):
        """ Plot Time Series """

        dfts = self.dfts
        axes = self.axes
        fig = self.fig
        lines = self.lines

        if self.spc_only:
            # for solar probe only
            try:
                if self.sc == 0:
                    ind = dfts['INSTRUMENT_FLAG'] == 1
                    dfts = self.dfts.copy()
                    dfts.loc[~ind, 
                    ['np','Vth','Vr','Vt','Vn','V','sigma_c','sigma_r','vbangle','di','rho_ci','beta','vsw','tadv']
                    ] = np.nan
            except:
                pass
        
        # particle mode
        try:
            parmode = self.dfts_misc['particle_mode']
        except:
            parmode = 'None'

        """make title"""
        fig.suptitle(
            "%s to %s, DEN=%s, MOM=%s, rolling_rate = %s" %(str(self.start_time), str(self.end_time), self.par_settings['density'], self.par_settings['moments'], self.rolling_rate)
            + "\n"
            + "SpaceCraft: %d, Resample Rate: %s, Window Option: %s (key=%d), MAG Frame: %d, Normed Mag : %d" %(self.sc, self.resample_rate, self.length_list[self.length_key], self.length_key, self.mag_option['sc'], self.mag_option['norm']),
            #+ "\n"
            #+ "Real Window Size = %s, Current Session (%s - %s)" %(str(self.end_time-self.start_time), self.start_time_0, self.end_time_0),
            fontsize = 'xx-large', horizontalalignment = 'left', x = 0.02
        )


        """magnetic field"""
        # self.mag_option = {norm: normalized with r^2, sc: 0 for RTN, 1 for SC}
        try:
            if not(update):
                # initialization
                ax = axes['mag']
                if self.mag_option['norm'] == 0:
                    if self.mag_option['sc'] == 0:
                        if self.mag_res is not None:
                            self.dfmag_raw_high_res[['Bx','By','Bz','Btot']].resample(self.mag_res).mean().plot(ax = ax, legend=False, style=['C0','C1','C2','k'], lw = 0.8, alpha = 0.7)
                        else:
                            dfts[['Br','Bt','Bn','B_RTN']].plot(ax = ax, legend=False, style=['C0','C1','C2','k'], lw = 0.8, alpha = 0.7)
                        ax.legend(['Br [nT]','Bt','Bn','|B|_RTN'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dfts[['Bx','By','Bz','B_SC']].plot(ax = ax, legend=False, style=['C0','C1','C2','k'], lw = 0.8, alpha = 0.7)
                        ax.legend(['Bx [nT]','By','Bz','|B|_SC'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_SC'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                elif self.mag_option['norm'] == 1:
                    if self.mag_option['sc'] == 0:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br'] = self.dfts['Br'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt'] = self.dfts['Bt'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn'] = self.dfts['Bn'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_RTN'] = (dftemp[['Br','Bt','Bn']]**2).sum(axis=1).apply(np.sqrt)
                        dftemp[['Br','Bt','Bn','B_RTN']].plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend([r'$Br*a(R)^2\ [nT]$',r'$Bt*a(R)\ a(R)=R/0.1AU$',r'$Bn*a(R)$',r'$|B|_{RTN}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Bx'] = self.dfts['Bx'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['By'] = self.dfts['By'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bz'] = self.dfts['Bz'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_SC'] = (dftemp[['Bx','By','Bz']]**2).sum(axis=1).apply(np.sqrt)
                        dftemp[['Bx','By','Bz','B_SC']].plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend([r'$Bx*a(R)^2\ [nT]$',r'$By*a(R)\ a(R)=R/0.1AU$',r'$Bz*a(R)$',r'$|B|_{SC}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_SC'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                else:
                    raise ValueError("mag_option['norm']==%d not supported!" %(self.mag_option['norm']))

                # aesthetic
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([-lim, lim])
                lines['mag'] = ax.get_lines()
            else:
                # update
                ax = axes['mag']
                ls = lines['mag']
                if self.mag_option['norm'] == 0:
                    if self.mag_option['sc'] == 0:
                        ls[0].set_data(dfts['Br'].index, dfts['Br'].values)
                        ls[1].set_data(dfts['Bt'].index, dfts['Bt'].values)
                        ls[2].set_data(dfts['Bn'].index, dfts['Bn'].values)
                        ls[3].set_data(dfts['B_RTN'].index, dfts['B_RTN'].values)
                        ax.legend(['Br [nT]','Bt','Bn','|B|_RTN'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        ls[0].set_data(dfts['Bx'].index, dfts['Bx'].values)
                        ls[1].set_data(dfts['By'].index, dfts['By'].values)
                        ls[2].set_data(dfts['Bz'].index, dfts['Bz'].values)
                        ls[3].set_data(dfts['B_SC'].index, dfts['B_SC'].values)
                        ax.legend(['Bx [nT]','By','Bz','|B|_SC'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_SC'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                elif self.mag_option['norm'] == 1:
                    if self.mag_option['sc'] == 0:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br'] = self.dfts['Br'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt'] = self.dfts['Bt'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn'] = self.dfts['Bn'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_RTN'] = (dftemp[['Br','Bt','Bn']]**2).sum(axis=1).apply(np.sqrt)
                        ls[0].set_data(dftemp['Br'].index, dftemp['Br'].values)
                        ls[1].set_data(dftemp['Bt'].index, dftemp['Bt'].values)
                        ls[2].set_data(dftemp['Bn'].index, dftemp['Bn'].values)
                        ls[3].set_data(dftemp['B_RTN'].index, dftemp['B_RTN'].values)
                        ax.legend([r'$Br*a(R)^2\ [nT]$',r'$Bt*a(R)\ a(R)=R/0.1AU$',r'$Bn*a(R)$',r'$|B|_{RTN}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Bx'] = self.dfts['Bx'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['By'] = self.dfts['By'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bz'] = self.dfts['Bz'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_SC'] = (dftemp[['Bx','By','Bz']]**2).sum(axis=1).apply(np.sqrt)
                        ls[0].set_data(dftemp['Bx'].index, dftemp['Bx'].values)
                        ls[1].set_data(dftemp['By'].index, dftemp['By'].values)
                        ls[2].set_data(dftemp['Bz'].index, dftemp['Bz'].values)
                        ls[3].set_data(dftemp['B_SC'].index, dftemp['B_SC'].values)
                        ax.legend([r'$Bx*a(R)^2\ [nT]$',r'$By*a(R)\ a(R)=R/0.1AU$',r'$Bz*a(R)$',r'$|B|_{SC}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_SC'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                else:
                    raise ValueError("mag_option['norm']==%d not supported!" %(self.mag_option['norm']))
                # aesthetic
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([-lim, lim])
                lines['mag'] = ax.get_lines()

                collect()
        except:
            raise ValueError('ss')
            print("MAG Plotting have some problems...")
            pass

        """vsw"""
        if not(update):
            try:
                ax = axes['vsw']
                # speeds
                dfts[['vsw']].plot(ax = ax, legend=False, style=['C2'], lw = 0.8)
                ax.legend(['Vsw[km/s]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                if dfts[['vsw']].max() < 700:
                    ax.set_ylim([100,700])
                else:
                    ax.set_ylim([300,900])
                lines['vsw'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['vsw']
                ls = lines['vsw']
                # speeds
                ls[0].set_data(dfts['V'].index, dfts['V'].values)
                ax.legend(['Vsw[km/s]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                if dfts[['vsw']].max() < 700:
                    ax.set_ylim([100,700])
                else:
                    ax.set_ylim([300,900])
                lines['vsw'] = ax.get_lines()
            except:
                pass

        """vth"""
        if not(update):
            try:
                ax = axes['vsw'].twinx()
                # speeds
                dfts[['Vth']].plot(ax = ax, legend=False, style=['C1'], lw = 0.6, alpha = 0.6)
                ax.legend(['Vth[km/s]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(dfts['Vth']),1.05*np.nanmax(dfts['Vth'])])
                lines['vth'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['vth']
                ls = lines['vth']
                # speeds
                ls[0].set_data(dfts['Vth'].index, dfts['Vth'].values)
                ax.legend(['Vth[km/s]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(dfts['Vth']),1.05*np.nanmax(dfts['Vth'])])
                lines['vth'] = ax.get_lines()
            except:
                pass

        

        """speeds"""
        if not(update):
            try:
                ax = axes['speed']
                # speeds
                dfts[['V','Vth']].plot(ax = ax, legend=False, style=['C1','C2'], lw = 0.8)
                ax.legend(['Vsw[km/s]','Vth[km/s]'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                if dfts['V'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                try:
                    min1 = np.nanmin(dfts['V']); max1 = np.nanmax(dfts['V'])
                    min2 = np.nanmin(dfts['Vth']); max2 = np.nanmax(dfts['Vth'])
                    min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                    ax.set_ylim([0.95*min0, 1.05*max0])
                except:
                    if verbose: print("Setting limit for speed failed! omitting...")
                lines['speed'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['speed']
                ls = lines['speed']
                # speeds
                ls[0].set_data(dfts['V'].index, dfts['V'].values)
                ls[1].set_data(dfts['Vth'].index, dfts['Vth'].values)
                ax.legend(['Vsw[km/s]','Vth[km/s]'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                if dfts['V'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                try:
                    min1 = np.nanmin(dfts['V']); max1 = np.nanmax(dfts['V'])
                    min2 = np.nanmin(dfts['Vth']); max2 = np.nanmax(dfts['Vth'])
                    min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                    ax.set_ylim([0.95*min0, 1.05*max0])
                except:
                    if verbose: print("Setting limit for speed failed! omitting...")
                lines['speed'] = ax.get_lines()
            except:
                pass

        """normalized parameters"""
        if not(update):
            try:
                ax = axes['norm']
                dfts[['sigma_c']].plot(ax = ax, legend=False, style=['C0'], lw = 0.9)
                # dfts[['sigma_r']].plot(ax = ax, legend=False, style=['C1'], lw = 0.7)
                # ax.legend([r'$\sigma_c$',r'$\sigma_r$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                ax.legend([r'$\sigma_c$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([-1.05,1.05])
                lines['norm'] = ax.get_lines()
            except:
                print("sigma_c initialization failed!")
        else:
            try:
                ax = axes['norm']
                ls = lines['norm']
                ls[0].set_data(dfts['sigma_c'].index, dfts['sigma_c'].values)
                # ls[1].set_data(dfts['sigma_r'].index, dfts['sigma_r'].values)
                # ax.legend([r'$\sigma_c$',r'$\sigma_r$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                ax.legend([r'$\sigma_c$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([-1.05,1.05])
                lines['norm'] = ax.get_lines()
            except:
                print("sigma_c update failed!")

        """carrington Longitude"""
        if not(update):
            try:
                ax = axes['norm'].twinx()
                axes['carr_lon'] = ax
                (dfts[['carr_lon']]).plot(ax = ax, legend=False, style=['C1'], lw = 0.8, alpha = 0.8)
                ax.legend(['Carr Lon'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                # ax.set_ylim([-185,185])
                # ax.set_yticks([-180,-90,0,90,180])
                lines['carr_lon'] = ax.get_lines()
            except:
                warnings.warn("carr long failed")
        else:
            try:
                ax = axes['carr_lon']
                ls = lines['carr_lon']
                # speeds
                ls[0].set_data(dfts['carr_lon'].index, (dfts[['carr_lon']]).values)
                ax.legend(['Carr Lon'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                # ax.set_ylim([-185,185])
                # ax.set_yticks([-180,-90,0,90,180])
                lines['carr_lon'] = ax.get_lines()
            except:
                warnings.warn('Carr long failed!')

        """magnetic compressibility"""
        # if not(update):
        #     try:
        #         ax = axes['norm'].twinx()
        #         axes['mag_compressibility'] = ax
        #         dfmag = self.dfmag_high_res
        #         Btot = dfmag['Btot']
        #         ind = (Btot.index >= self.start_time) & (Btot.index <= self.end_time)
        #         Btot = Btot[ind]

        #         mag_coms = {}
        #         mckeys = ['1000s','3600s', '10000s']
        #         for k in mckeys:
        #             mag_coms[k] = ((Btot.rolling(k).std())/(Btot.rolling(k).mean())).resample(self.resample_rate).mean()

        #         self.mag_coms = mag_coms

        #         for i1 in range(len(mckeys)):
        #             k = mckeys[i1]
        #             mag_coms[k].plot(ax = ax, style = ['C%d'%(i1+1)], lw = 0.8, alpha = 0.8)

        #         ax.set_yscale('log')
        #         ax.set_ylim([10**(-2.05), 10**(0.05)])
        #         ax.axhline(y = 1e-2, color = 'gray', ls = '--', lw = 0.6)
        #         ax.axhline(y = 1e-1, color = 'gray', ls = '--', lw = 0.6)
        #         ax.axhline(y = 1e0, color = 'gray', ls = '--', lw = 0.6)

        #         ax.legend([r'$10^{-3}$ Hz',r'$1H^{-1}$ Hz',r'$10^{-4}$ Hz'], fontsize='medium', frameon=False, bbox_to_anchor=(1.02,0.8), loc = 2)
        #         ax.set_xticks([], minor=True)
        #         ax.set_xticks([])
        #         ax.set_xlabel('')
        #         ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
        #         lines['mag_compressibility'] = ax.get_lines()
        #     except:
        #         raise ValueError("mag compressibility initialization failed!")
        # else:
        #     try:
        #         ax = axes['mag_compressibility']
        #         ls = lines['mag_compressibility']
        #         if (self.sc == 0) | (self.sc == 1):
        #             # for psp and solo, the raw resolution is 5s
        #             Btot = (self.dfts_raw0[['Br','Bt','Bn']]**2).sum(axis = 1)
        #         elif (self.sc == 2) | (self.sc == 3) | (self.sc == 4):
        #             # for Helios-1/2 and Ulysses, load high-res mag data for this line
        #             dfmag, infos = LoadHighResMagWrapper(self.sc, self.start_time, self.end_time, credentials = self.credentials)
        #             Btot = dfmag['Btot']
        #         else:
        #             pass
        #         ind = (Btot.index >= self.start_time) & (Btot.index <= self.end_time)
        #         Btot = Btot[ind]

        #         mag_coms = {}
        #         mckeys = ['1000s','3600s','10000s']
        #         for k in mckeys:
        #             mag_coms[k] = ((Btot.rolling(k).std())/(Btot.rolling(k).mean())).resample(self.resample_rate).mean()

        #         self.mag_coms = mag_coms

        #         for i1 in range(len(mckeys)):
        #             k = mckeys[i1]
        #             ls[i1].set_data(mag_coms[k].index, mag_coms[k].values)

        #         ax.axhline(y = 1e-2, color = 'gray', ls = '--', lw = 0.6)
        #         ax.axhline(y = 1e-1, color = 'gray', ls = '--', lw = 0.6)
        #         ax.axhline(y = 1e0, color = 'gray', ls = '--', lw = 0.6)

        #         ax.legend([r'$10^{-3}$ Hz',r'$1H^{-1}$ Hz',r'$10^{-4}$ Hz'], fontsize='medium', frameon=False, bbox_to_anchor=(1.02,0.8), loc = 2)
        #         ax.set_xticks([], minor=True)
        #         ax.set_xticks([])
        #         ax.set_xlabel('')
        #         ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #         lines['mag_compressibility'] = ax.get_lines()

        #     except:
        #         print("mag compressibility update failed!")




        """density"""
        if not(update):
            try:
                ax = axes['density']
                # speeds
                dfts[['np']].plot(ax = ax, legend=False, style=['k'], lw = 0.8, alpha = 0.7)
                ax.legend(['np[$cm^{-3}$]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(dfts['np']),1.05*np.nanmax(dfts['np'])])
                lines['density'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['density']
                ls = lines['density']
                # speeds
                ls[0].set_data(dfts['np'].index, dfts['np'].values)
                ax.legend(['np[$cm^{-3}$]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(dfts['np']),1.05*np.nanmax(dfts['np'])])
                lines['density'] = ax.get_lines()
            except:
                pass

        """na/np"""
        if not(update):
            try:
                ax = axes['density'].twinx()
                self.axes['nanp_ratio'] = ax
                dfts[['nanp_ratio']].plot(ax = ax, legend=False, style=['C3--'], lw = 0.8, alpha = 0.6)
                ax.legend([r'na/np'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                try:
                    ax.set_ylim([-0.005, 0.08])
                except:
                    pass
                lines['nanp_ratio'] = ax.get_lines()
            except:
                warnings.warn("Initializing nanp_ratio failed!")
        else:
            try:
                ax = axes['nanp_ratio']
                ls = lines['nanp_ratio']
                ls[0].set_data(dfts['nanp_ratio'].index, dfts['nanp_ratio'].values)
                ax.legend([r'na/np'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                try:
                    ax.set_ylim([-0.005, 0.08])
                except:
                    pass
                lines['nanp_ratio'] = ax.get_lines()
            except:
                warnings.warn("Updating nanp_ratio failed!...")

        # """malfven"""
        # if not(update):
        #     try:
        #         ax = axes['density'].twinx()
        #         self.axes['malfven'] = ax
        #         dfts[['malfven']].plot(ax = ax, legend=False, style=['C3--'], lw = 0.8, alpha = 0.6)
        #         ax.legend([r'$M_{A}$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
        #         # ax.set_xticks([], minor=True)
        #         # ax.set_xticks([])
        #         # ax.set_xlabel('')
        #         # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
        #         ax.axhline(y=1.0, ls = '--', lw = 1.5, color = 'darkgreen')
        #         lines['malfven'] = ax.get_lines()
        #         ax.set_yscale('log')
        #         ax.set_ylim([10**(-1.1), 10**(1.1)])
        #     except:
        #         warnings.warn("Initializing nanp_ratio failed!")
        # else:
        #     try:
        #         ax = axes['malfven']
        #         ls = lines['malfven']
        #         ls[0].set_data(dfts['malfven'].index, dfts['malfven'].values)
        #         ax.legend([r'$M_{A}$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
        #         # ax.set_xticks([], minor=True)
        #         # ax.set_xticks([])
        #         # ax.set_xlabel('')
        #         # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
        #         lines['malfven'] = ax.get_lines()
        #         ax.set_yscale('log')
        #         ax.set_ylim([10**(-1.1), 10**(1.1)])
        #     except:
        #         warnings.warn("Updating malfven failed!...")

        # """scales"""
        # if not(update):
        #     try:
        #         ax = axes['density'].twinx()
        #         self.axes['scale'] = ax
        #         dfts[['di','rho_ci']].plot(ax = ax, legend=False, style=['C3--','C4--'], lw = 0.8, alpha = 0.6)
        #         ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
        #         if dfts['di'].apply(np.isnan).sum() != len(dfts):
        #             ax.set_yscale('log')
        #         ax.set_xticks([], minor=True)
        #         ax.set_xticks([])
        #         ax.set_xlabel('')
        #         ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #         try:
        #             min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
        #             min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
        #             min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
        #             ax.set_ylim([0.95*min0, 1.05*max0])
        #         except:
        #             if verbose: print("Setting scale limit failed... omitting...")
        #         lines['scale'] = ax.get_lines()
        #     except:
        #         raise ValueError("Initializing scale failed!")
        # else:
        #     try:
        #         ax = axes['scale']
        #         ls = lines['scale']
        #         ls[0].set_data(dfts['di'].index, dfts['di'].values)
        #         ls[1].set_data(dfts['rho_ci'].index, dfts['rho_ci'].values)
        #         ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
        #         if dfts['di'].apply(np.isnan).sum() != len(dfts):
        #             ax.set_yscale('log')
        #         ax.set_xticks([], minor=True)
        #         ax.set_xticks([])
        #         ax.set_xlabel('')
        #         ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #         try:
        #             min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
        #             min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
        #             min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
        #             ax.set_ylim([0.95*min0, 1.05*max0])
        #         except:
        #             if verbose: print("Setting scale limit failed... omitting...")
        #         lines['scale'] = ax.get_lines()
        #     except:
        #         raise ValueError("Updating scale failed!...")

        # """spectrogram"""
        # try:
        #     ax = axes['spec']
        #     res_r = ax.specgram(dfts_raw['Br']-dfts_raw['Br0'],Fs=1,NFFT=2**13, noverlap=2**6)
        #     res_t = ax.specgram(dfts_raw['Bt']-dfts_raw['Bt0'],Fs=1,NFFT=2**13, noverlap=2**6)
        #     res_n = ax.specgram(dfts_raw['Bn']-dfts_raw['Bn0'],Fs=1,NFFT=2**13, noverlap=2**6)
        #     ax.cla()
        #     t = (dfts_raw.index[res_r[2].astype('int')])
        #     self.spec_res = res_r
        #     self.spec_x_ind = res_r[2].astype('int')
        #     self.spec_x_axis = t
        #     im = ax.pcolormesh(t, res_r[1][1:], np.log10(res_r[0]+res_t[0]+res_n[0])[1:,:], shading='gouraud')#, vmin=-2, vmax=4)
        #     ax.set_yscale('log')
        #     ax.set_xticks([], minor=True)
        #     ax.set_xticks([])
        #     ax.set_xlabel('')
        #     ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #     axins = inset_axes(ax,
        #            width="2%",  # width = 5% of parent_bbox width
        #            height="100%",  # height : 50%
        #            loc='lower left',
        #            bbox_to_anchor=(1.01, 0., 1, 1),
        #            bbox_transform=ax.transAxes,
        #            borderpad=0,
        #            )
        #     plt.colorbar(im, ax = ax, cax = axins)
        # except:
        #     pass

        """angles"""
        if not(update):
            try:
                ax = axes['ang']
                try:
                    dfts[['vbangle','brangle']].plot(ax = ax, legend=False, style = ['r--','b--'], lw = 0.8)
                    ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    dfts[['vbangle']].plot(ax = ax, legend=False, style = ['r--','b--'], lw = 0.8)
                    ax.legend([r'$<\vec V, \vec B>$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)

                # plot latitude
                try:
                    (dfts['lat']+90).plot(ax = ax, legend = False, style = 'm--', lw = 1.2)
                    ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$', 'Lat+90'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    pass

                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.grid(False)
                ax.set_ylim([-5,185])
                ax.set_yticks([0,45,90,135,180])
                lines['ang'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['ang']
                ls = lines['ang']
                try:
                    ls[0].set_data(dfts['vbangle'].index, dfts['vbangle'].values)
                    ls[1].set_data(dfts['brangle'].index, dfts['brangle'].values)
                    ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    ls[0].set_data(dfts['vbangle'].index, dfts['vbangle'].values)
                    ax.legend([r'$<\vec V, \vec B>$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                
                try:
                    if len(ls) == 3:
                        ls[2].set_data(dfts['lat'].index, (dfts['lat']+90))
                        ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$', 'Lat+90'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                    elif len(ls) == 2:
                        ls[1].set_data(dfts['lat'].index, (dfts['lat']+90))
                        ax.legend([r'$<\vec r, \vec B>$', 'Lat+90'], fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    pass

                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.grid(False)
                ax.set_ylim([-5,185])
                ax.set_yticks([0,45,90,135,180])
                lines['ang'] = ax.get_lines()
            except:
                pass            
        
        """plasma beta"""
        if not(update):
            try:
                ax = axes['ang'].twinx()
                dfts['beta'].plot(ax = ax, style = ['k-'], legend='False', lw = 1.0)
                ax.legend([r'$\beta$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                if dfts['beta'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                minbeta = np.abs(np.log10(np.nanmin(dfts['beta'])))
                maxbeta = np.abs(np.log10(np.nanmax(dfts['beta'])))
                lim = np.max([minbeta,maxbeta])
                ax.set_yticks([1e-2,1e-1,1e0,1e1,1e2])
                ax.set_ylim([10**(-lim),10**(lim)])
                lines['beta'] = ax.get_lines()
                axes['beta'] = ax
                ax.axhline(1, color = 'gray', ls = '--', lw = 0.6)
            except:
                pass
        else:
            try:
                ax = axes['beta']
                ls = lines['beta']
                ls[0].set_data(dfts['beta'].index, dfts['beta'].values)
                ax.legend([r'$\beta$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                if dfts['beta'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                # ax.set_xticks([], minor=True)
                # ax.set_xticks([])
                # ax.set_xlabel('')
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                minbeta = np.abs(np.log10(np.nanmin(dfts['beta'])))
                maxbeta = np.abs(np.log10(np.nanmax(dfts['beta'])))
                lim = np.max([minbeta,maxbeta])*1.05
                ax.set_yticks([1e-2,1e-1,1e0,1e1,1e2])
                ax.set_ylim([10**(-lim),10**(lim)])
                lines['beta'] = ax.get_lines()
            except:
                pass

        # """alfvenicity"""
        # if not(update):
        #     try:
        #         ax = axes['alfvenicity']
        #         dfts['alfvenicity'].apply(np.abs).plot(ax = ax, style = ['C0-'], legend=False, lw = 1.0)
        #         ax.legend([r'$\delta |\vec B|/|\delta\vec B|$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
        #         ax.set_yscale('log')
        #         ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #         lines['alfvenicity'] = ax.get_lines()
        #     except:
        #         pass
        # else:
        #     try:
        #         ax = axes['alfvenicity']
        #         ls = lines['alfvenicity']
        #         ls[0].set_data(dfts['alfvenicity'].index, dfts['alfvenicity'].apply(np.abs).values)
        #         ax.legend([r'$\delta |\vec B|/|\delta\vec B|$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
        #         ax.set_yscale('log')
        #         ax.set_xlim([dfts.index[0], dfts.index[-1]])
        #         lines['alfvenicity'] = ax.get_lines()
        #     except:
        #         pass

        """rau"""
        if not(update):
            try:
                ax = axes['rau']
                ydata = dfts['Dist_au']*au_to_rsun
                (ydata).plot(ax = ax, style = ['k'], legend = False, lw = 0.8)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                (ydata).plot(ax = ax, style = ['k'], legend = False, lw = 0.8)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
        else:
            try:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['Dist_au']*au_to_rsun
                ls[0].set_data(dfts['Dist_au'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                ls[0].set_data(dfts['RAD_AU'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()

        """tadv"""
        if not(update):
            try:
                ax = axes['rau'].twinx()
                axes['tadv'] = ax
                dfts['tadv'].plot(ax = ax, style = ['r'], legend = False, lw = 0.8)
                ax.legend([r'$\tau_{adv}\ [Hr]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([np.nanmin(dfts['tadv'])*0.95, np.nanmax(dfts['tadv'])*1.05])
                lines['tadv'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['tadv']
                ls = lines['tadv']
                ls[0].set_data(dfts['tadv'].index, dfts['tadv'].values)
                ax.legend([r'$\tau_{adv}\ [Hr]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
                ax.set_ylim([np.nanmin(dfts['tadv'])*0.95, np.nanmax(dfts['tadv'])*1.05])
                lines['tadv'] = ax.get_lines()
            except:
                pass

        """import time series"""
        try:
            import_dict = self.import_timeseries
            imkey = import_dict['key']
            imdf = import_dict['df']
            styles = import_dict['styles']
            labels = import_dict['labels']
            cols = import_dict['columns']
            if not(update):
                try:
                    ax = axes[imkey]
                    imdf[cols].plot(ax = ax, style = styles, lw = 0.8, legend=False)
                    ax.legend(labels, fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                    if 'ylim' in import_dict.keys():
                        ax.set_ylim(import_dict['ylim'])
                    if 'yscale' in import_dict.keys():
                        ax.set_yscale(import_dict['yscale'])
                    lines[imkey] = ax.get_lines()
                    if 'axhline' in import_dict.keys():
                        ax.axhline(y=import_dict['axhline']['y'], color=import_dict['axhline']['color'], ls = import_dict['axhline']['ls'], lw = import_dict['axhline']['lw'])
                except:
                    raise ValueError('...')
            else:
                try:
                    ax = axes[imkey]
                    ls = lines[imkey]
                    for i1 in range(len(ls)):
                        line = ls[i1]
                        line.set_data(imdf[cols[i1]].index, imdf[cols[i1]].values)
                    ax.legend(labels, fontsize='large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                    if 'ylim' in import_dict.keys():
                        ax.set_ylim(import_dict['ylim'])
                    if 'yscale' in import_dict.keys():
                        ax.set_yscale(import_dict['yscale'])
                    lines[imkey] = ax.get_lines()
                except:
                    pass

            """import time series twinx"""
            if 'twinx' in self.import_timeseries.keys():
                twinx_dict = self.import_timeseries['twinx']
                imkey_mother = self.import_timeseries['key']
                imkey = twinx_dict['key']
                imdf = twinx_dict['df']
                styles = twinx_dict['styles']
                labels = twinx_dict['labels']
                cols = twinx_dict['columns']
                if not(update):
                    try:
                        ax = axes[imkey_mother].twinx()
                        axes[imkey] = ax
                        imdf[cols].plot(ax = ax, style = styles, lw = 0.8, legend=False)
                        ax.legend(labels, fontsize='large', frameon=False, bbox_to_anchor=(1.02,0), loc = 3)
                        if 'ylim' in twinx_dict.keys():
                            ax.set_ylim(twinx_dict['ylim'])
                        if 'yscale' in twinx_dict.keys():
                            ax.set_yscale(twinx_dict['yscale'])
                        lines[imkey] = ax.get_lines()
                        if 'axhline' in twinx_dict.keys():
                            ax.axhline(y=twinx_dict['axhline']['y'], color=twinx_dict['axhline']['color'], ls = twinx_dict['axhline']['ls'], lw = twinx_dict['axhline']['lw'])
                    except:
                        pass
                else:
                    try:
                        ax = axes[imkey]
                        ls = lines[imkey]
                        for i1 in range(len(ls)):
                            line = ls[i1]
                            line.set_data(imdf[cols[i1]].index, imdf[cols[i1]].values)
                        ax.legend(labels, fontsize='large', frameon=False, bbox_to_anchor=(1.02,0), loc = 3)
                        if 'ylim' in twinx_dict.keys():
                            ax.set_ylim(twinx_dict['ylim'])
                        if 'yscale' in twinx_dict.keys():
                            ax.set_yscale(twinx_dict['yscale'])
                        lines[imkey] = ax.get_lines()
                    except:
                        pass
        except:
            pass


        # aesthetics
        for i1 in range(len(self.axeslist)):
            if i1 != len(self.axeslist)-1:
                ax = self.axeslist[i1]
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')

            # ax.set_xlim([dfts.index[0].timestamp(), dfts.index[-1].timestamp()])
            ax.set_xlim([dfts.index[0], dfts.index[-1]])

        # fig.tight_layout()
        self.lines = lines
        self.axes = axes
        collect()

        

    def ChangeWindowSize(self, key):
        """ change the window size with number key """
        self.length_key = key
        self.window_size_histories.append(
            {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
        )
        print("Current Window Length is: %s" %(self.length_list[self.length_key]))
        tmid = self.start_time + 0.5 * (self.end_time - self.start_time)
        self.start_time = tmid - pd.Timedelta(self.length_list[self.length_key])*0.5
        self.end_time = tmid + pd.Timedelta(self.length_list[self.length_key])*0.5
        self.CheckTimeBoundary()
        self.UpdateFigure(
            self.start_time, self.end_time
        )


    def FindSelectedIntervals(self, x):
        flags = np.zeros_like(self.selected_intervals)
        for i1 in range(len(self.selected_intervals)):
            si = self.selected_intervals[i1]
            if (x > si['start_time']) & (x < si['end_time']):
                flags[i1] = 1
        
        inds = np.where(flags)
        can_si = (np.array(self.selected_intervals))[inds]
        lengths = np.array([np.abs(can_si[i1]['end_time']-can_si[i1]['start_time']) for i1 in range(len(can_si))])
            
        return inds[0][np.argmin(lengths)]
            

    #--------  Visual Interface --------#

    def on_key_event(self, event):
        """ On Key Event """

        # ------ character key for functions ------ #
        if (event.key == 'd'):
            try:
                x = self.current_x
                i1 = self.FindSelectedIntervals(x)
                selected_interval = self.selected_intervals[i1]

                if 'QualityFlag' in selected_interval.keys():
                    flag = selected_interval['QualityFlag'] 

                # remove the red vline and shaded area
                for k, ax in self.axes.items():
                    selected_interval['rects'][k].remove()
                    selected_interval['lines1'][k].remove()
                    selected_interval['lines2'][k].remove()
                    ax.figure.canvas.draw()
                # pop the instance
                self.selected_intervals.pop(i1)
            except:
                pass
        elif (event.key == 'backspace'):
            """go back to the last frame"""
            try:
                window_size_history = self.window_size_histories.pop()
                self.start_time = window_size_history['start_time']
                self.end_time = window_size_history['end_time']
                self.sc = window_size_history['sc']
                self.resample_rate = window_size_history['resample_rate']
                self.UpdateFigure(
                    self.start_time, self.end_time
                )
            except:
                raise ValueError("event key: %s failed!" %(event.key))
                pass
        elif (event.key == 'p'):
            try:
                x = self.current_x
                self.p_application(x)
            except:
                raise ValueError("Func p: failed!")
            collect()
            pass
        # ------ b/n for changing magnetic field options ------ #  
        elif (event.key == 'b'):
            # use rtn (sc = 0), sc (sc = 1)
            if self.mag_option['sc'] == 0:
                self.mag_option['sc'] = 1
            elif self.mag_option['sc'] == 1:
                self.mag_option['sc'] = 0
            else: raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
            self.PlotTimeSeries(update=True)
            pass
        elif (event.key == 'n'):
            # original (norm = 0), normalized (norm = 1)
            if self.mag_option['norm'] == 0:
                self.mag_option['norm'] = 1
            elif self.mag_option['norm'] == 1:
                self.mag_option['norm'] = 0
            else: raise ValueError("mag_option['norm']==%d not supported!" %(self.mag_option['norm']))
            self.PlotTimeSeries(update=True)
            pass
        # ------ i(nstrument) for changing spc only ------ # 
        elif (event.key == 'i'):
            # default self.spc_only = True
            if self.spc_only == True:
                self.spc_only = False
            else:
                self.spc_only = True
            self.PlotTimeSeries(update=True)
            pass
        # ------ left right for navigate ------ #
        elif (event.key == 'right'):
            self.window_size_histories.append(
                {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
            )
            dt = (self.end_time - self.start_time)*0.5
            self.start_time = self.start_time + dt
            self.end_time = self.end_time + dt
            self.CheckTimeBoundary()
            self.UpdateFigure(
                self.start_time, self.end_time
            )
        elif (event.key == 'left'):
            self.window_size_histories.append(
                {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
            )
            dt = (self.end_time - self.start_time)*0.5
            self.start_time = self.start_time - dt
            self.end_time = self.end_time - dt
            self.CheckTimeBoundary()
            self.UpdateFigure(
                self.start_time, self.end_time
            )
        # ------ up down for resolution ------ #
        elif (event.key == 'down'):
            self.window_size_histories.append(
                {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
            )
            if self.resample_rate_key < len(self.resample_rate_list)-1:
                self.resample_rate_key += 1
            else: pass
            print("Resample Rate: %s" %(self.resample_rate))
            self.resample_rate = self.resample_rate_list[self.resample_rate_key]
            self.CheckTimeBoundary()
            self.UpdateFigure(
                self.start_time, self.end_time,
                update_resample_rate_only = True
            )
        elif (event.key == 'up'):
            self.window_size_histories.append(
                {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
            )
            if self.resample_rate_key > 0:
                self.resample_rate_key -= 1
            else: pass
            print("Resample Rate: %s" %(self.resample_rate))
            self.resample_rate = self.resample_rate_list[self.resample_rate_key]
            self.CheckTimeBoundary()
            self.UpdateFigure(
                self.start_time, self.end_time,
                update_resample_rate_only = True
            )
        # ------ number key for interval length ------ #
        elif (event.key == '1'):
            self.ChangeWindowSize(0)
        elif (event.key == '2'):
            self.ChangeWindowSize(1)
        elif (event.key == '3'):
            self.ChangeWindowSize(2)
        elif (event.key == '4'):
            self.ChangeWindowSize(3)
        elif (event.key == '5'):
            self.ChangeWindowSize(4)
        elif (event.key == '6'):
            self.ChangeWindowSize(5)
        elif (event.key == '7'):
            self.ChangeWindowSize(6)
        elif (event.key == '8'):
            self.ChangeWindowSize(7)
        elif (event.key == '9'):
            self.ChangeWindowSize(8)
        elif (event.key == '0'):
            self.ChangeWindowSize(9)
        else:
            pass


    def on_click_event(self, event):
        """ click event """
        if event.button is MouseButton.RIGHT:
            """left click to select event"""
            x, y = event.xdata, event.ydata
            # print(x,y)
            # red vertical line instance
            red_vline = {
                'timestamp': pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp(),
                'x_period': int(x),
                'lines': {}
                }
            for k, ax in self.axes.items():
                red_vline['lines'][k] = ax.axvline(x, color = 'r', ls = '--', lw = 2)
            
            self.red_vlines.append(red_vline)

            # show shaded area between the last two red vline
            if (np.mod(len(self.red_vlines),2) == 0) & (len(self.red_vlines) >= 2):
                l1 = self.red_vlines[-1]
                l2 = self.red_vlines[-2]
                tstart = np.min([l1['timestamp'], l2['timestamp']])
                tend = np.max([l1['timestamp'], l2['timestamp']])
                # selected interval instance
                selected_interval = {
                    'spacecraft': self.sc,
                    'start_time': tstart,
                    'end_time': tend,
                    'TimeSeries': None,
                    'rects': {},
                    'lines1': l1['lines'],
                    'lines2': l2['lines']
                }
                for k, ax in self.axes.items():
                    selected_interval['rects'][k] = ax.axvspan(l1['timestamp'].timestamp(), l2['timestamp'].timestamp(), alpha = 0.02, color = 'red')

                self.selected_intervals.append(selected_interval)

            self.fig.canvas.draw()
        elif event.button is MouseButton.LEFT:
            """right click to zoom in"""
            x = event.xdata
            green_vline = {
                'timestamp': pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp(),
                'x_period': int(x),
                'lines': {}
            }
            for k, ax in self.axes.items():
                green_vline['lines'][k] = ax.axvline(x, color = 'g', ls = '--', lw = 2)

            self.green_vlines.append(green_vline)

            # zoom in if we have two green line
            try:
                if (len(self.green_vlines)==2):
                    l1 = self.green_vlines[0]
                    l2 = self.green_vlines[1]
                    tstart = np.min([l1['timestamp'], l2['timestamp']])
                    tend = np.max([l1['timestamp'], l2['timestamp']])

                    # store the window size history
                    self.window_size_histories.append(
                    {'sc':self.sc, 'start_time': self.start_time, 'end_time': self.end_time, 'resample_rate': self.resample_rate}
                    )

                    # set current window start_time and end_time
                    self.start_time = tstart
                    self.end_time = tend
                    self.UpdateFigure(
                    self.start_time, self.end_time
                    )
                    for k, ax in self.axes.items():
                        l1['lines'][k].remove()
                        l2['lines'][k].remove()
                        ax.figure.canvas.draw()
                    
                    # empty the stack
                    self.green_vlines = []
            except:
                raise ValueError("Zooming failed!")

            collect()


    def set_cross_hair_visible(self, visible):
        need_redraw = self.vertical_lines['mag'].get_visible() != visible
        for k, _ in self.axes.items():
            # self.horizontal_lines[k].set_visible(visible)
            self.vertical_lines[k].set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw


    def on_mouse_move(self, event):
        # ax = self.axes['mag']
        self.eventx = event.xdata
        self.eventy = event.ydata
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(True)
            if need_redraw:
                for _, ax in self.axes.items():
                    ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            self.current_x = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp()
            # self.current_x = pd.Period(ordinal=int(event.xdata), freq='5s').to_timestamp()
            # update the line positions

            # find rau, tadv and calculate the recommended t interval
            # matplotlib and pandas does not work well together
            # solution: https://stackoverflow.com/questions/54035067/matplotlib-event-xdata-out-of-timeries-range?noredirect=1&lq=1
            xt = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp()
            # xt = xt = pd.Period(ordinal=int(event.xdata), freq='5s').to_timestamp()
            self.xt = xt
            # try:
            tind = np.where(np.abs(self.dfts.index - xt) < 2*pd.Timedelta(self.resample_rate))[0][0]
            # tind = np.where(np.abs(self.dfts.index - xt) < pd.Timedelta('5s'))[0][0]
            vsw = self.dfts['vsw'][tind]
            rau = self.dfts['Dist_au'][tind]
            if np.isnan(rau):
                try:
                    rau = self.dfts['RAD_AU'][tind]
                except:
                    pass

            t_len1 = SolarWindCorrelationLength(rau)
            
            try:
                tadv = rau/vsw * au_to_km / 3600
                t_len2 = SolarWindCorrelationLength(rau, Vsw = vsw, correction = True)
            except:
                tadv = np.nan
                # t_len1 = np.nan
                t_len2 = np.nan

            if np.mod(len(self.red_vlines),2) == 1:
                try:
                    dtrl = (xt - self.red_vlines[-1]['timestamp']).total_seconds()/3600
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU] / %.2f [Rs], tadv = %.2f [Hr], Selected = %.2f [Hr], vsw = %.0f [km/s]" %(xt, rau, rau*au_to_rsun, tadv, dtrl, vsw)
                    )
                except:
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU] / %.2f [Rs], tadv = %.2f [Hr], vsw = %.0f [km/s]" %(xt, rau, rau*au_to_rsun, tadv, vsw)
                    )
            else:
                try:
                    x = self.current_x
                    for i1 in range(len(self.selected_intervals)):
                        selected_interval = self.selected_intervals[i1]
                        if (x > selected_interval['start_time']) & (x < selected_interval['end_time']):
                            dtint = (selected_interval['end_time']- selected_interval['start_time']).total_seconds()/3600
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU] / %.2f [Rs], tadv = %.2f [Hr], Current = %.2f [Hr], vsw = %.0f [km/s]" %(xt, rau, rau*au_to_rsun, tadv, dtint, vsw)
                    )
                except:
                    # raise ValueError("..")
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU] / %.2f [Rs], tadv = %.2f [Hr], vsw = %.0f [km/s]" %(xt, rau, rau*au_to_rsun, tadv, vsw)
                    )
            # xt = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate)
            # self.text.set_text("%s" %(str(xt)))
            self.text.set_fontsize('x-large')
            self.text.set_x(0.0)
            self.text.set_y(1.05)

            for k, ax in self.axes.items():
                # self.horizontal_lines[k].set_ydata(y)
                self.vertical_lines[k].set_xdata([x])
                ax.figure.canvas.draw()
            

    def disconnect(self):
        """ Disconnect all callbacks """
        self.fig.canvas.mpl_disconnect(self.cid0)
        self.fig.canvas.mpl_disconnect(self.cid1)
        self.fig.canvas.mpl_disconnect(self.cid2)
        

    def connect(self):
        """ Connect to matplotlib """
        self.cid0 = self.fig.canvas.mpl_connect('button_press_event', self.on_click_event)
        self.cid1 = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.on_key_event)


    #-------- p application --------#

    def p_application(self, x, additional_settings = {}, check_exist=True, interval_index=None):

        if interval_index is None:
            i1 = self.FindSelectedIntervals(x)
        else:
            i1 = interval_index

        selected_interval = self.selected_intervals[i1]
        self.selected_interval = selected_interval
        print("%s, %s, len=%s" %(self.selected_interval['start_time'], self.selected_interval['end_time'], self.selected_interval['end_time']-self.selected_interval['start_time']))
        si = self.selected_intervals[i1]
        t0 = selected_interval['start_time']; t1 = selected_interval['end_time']

        # calculate the time series for later diagnostics
        if len(self.p_funcs) > 0:
            ind = (self.dfts.index > t0) & (self.dfts.index < t1)
            dftemp = self.dfts.loc[ind]

            # save the time series to dataframe
            si['TimeSeries'] = dftemp
            si['LTSWsettings'] = self.LTSWsettings

            # save dfmag
            ind = (self.dfmag_raw_high_res.index > t0) & (self.dfmag_raw_high_res.index <= t1)
            si['dfmag_raw'] = self.dfmag_raw_high_res[ind]
            ind = (self.dfmag_high_res.index > t0) & (self.dfmag_high_res.index <= t1)
            si['dfmag'] = self.dfmag_high_res[ind]

            # save dist_au to the dictionary
            Btot = si['dfmag']['Btot']
            Dist_au = self.dfts_raw0['Dist_au'].resample(pd.infer_freq(Btot.index)).interpolate()
            self.selected_interval['Dist_au_raw'] = Dist_au[Btot.index]
            r = Dist_au[Btot.index].values
            self.selected_interval['Dist_au'] = r

            
            if (self.sc == 0):
                si['par_settings'] = self.par_settings

            if 'rescale_mag' in self.p_funcs.keys():
                rescale_mag = self.p_funcs['rescale_mag']
            else:
                rescale_mag = False

            # load the magnetic field data and create empty diagnostics
            try:
                # import the preloaded high resolution magnetic field data to save time
                PreloadDiagnostics(
                    si, 
                    self.p_funcs, 
                    resolution = self.high_res_resolution, 
                    credentials = self.credentials,
                    import_dfmag = {
                        'dfmag_raw': si['dfmag_raw'],
                        'dfmag': si['dfmag'],
                        'infos': self.dfmag_infos
                    },
                    rescale_mag = rescale_mag,
                    check_exist = check_exist
                )
            except:
                raise ValueError("Import dfmag failed!")
                PreloadDiagnostics(
                    si, self.p_funcs, 
                    resolution = self.high_res_resolution, credentials = self.credentials,
                    rescale_mag = rescale_mag
                )

        print("Rmin=%.2f, Rmax=%.2f, Rmean=%.2f" %(selected_interval['TimeSeries']['Dist_au'].min()*au_to_rsun, selected_interval['TimeSeries']['Dist_au'].max()*au_to_rsun, selected_interval['TimeSeries']['Dist_au'].mean()*au_to_rsun))
        print("MAlfven_min=%.2f, MAlfven_max=%.2f, MAlfven_mean=%.2f" %(selected_interval['TimeSeries']['malfven'].min(), selected_interval['TimeSeries']['malfven'].max(), selected_interval['TimeSeries']['malfven'].mean()))

        # compare magnetic field rescaling
        if 'compare_mag_rescale' in self.p_funcs.keys():
            # calculate diagnostics without rescale
            PreloadDiagnostics(
                si, 
                self.p_funcs, 
                resolution = self.high_res_resolution, 
                credentials = self.credentials,
                import_dfmag = {
                    'dfmag_raw': si['dfmag_raw'],
                    'dfmag': si['dfmag'],
                    'infos': self.dfmag_infos
                },
                rescale_mag = False
            )
            freqs_wl1, PSD_wl1 = si['wavelet_PSD']['freqs'], si['wavelet_PSD']['PSD']

            # calculate diagnostics with rescale
            PreloadDiagnostics(
                si, 
                self.p_funcs, 
                resolution = self.high_res_resolution, 
                credentials = self.credentials,
                import_dfmag = {
                    'dfmag_raw': si['dfmag_raw'],
                    'dfmag': si['dfmag'],
                    'infos': self.dfmag_infos
                },
                rescale_mag = True
            )
            freqs_wl2, PSD_wl2 = si['wavelet_PSD']['freqs'], si['wavelet_PSD']['PSD']

            coi, scales = si['wavelet_PSD']['coi'], si['wavelet_PSD']['scales']
            freqs_FFT, PSD_FFT = si['wavelet_PSD']['freqs_FFT'], si['wavelet_PSD']['PSD_FFT']

            # show coi range, threshold = 80% under coi
            if 'coi_thresh' in self.p_funcs['wavelet_PSD'].keys():
                coi_thresh = self.p_funcs['wavelet_PSD']['coi_thresh']
            else:
                coi_thresh = 0.8
            ind = np.array([np.sum(s < coi)/len(coi) for s in scales]) < coi_thresh
            import_coi = {
                'range': [freqs_wl1[ind][0], freqs_wl1[ind][-1]],
                'label': r'outside COI > %.1f %%' %((1-coi_thresh)*100)
            }

            # plot the spectrum
            self.bpfg_wl = BPFG(
                    freqs_wl1, PSD_wl1, label = 'original',
                    secondary = {
                        'x': freqs_FFT,
                        'y': PSD_FFT,
                        'label': 'PSD_FFT'
                    },
                    third = {
                        'x': freqs_wl2,
                        'y': PSD_wl2,
                        'label': 'rescaled'
                    },
                    import_coi = import_coi
                    )
            self.bpfg_wl.connect()


        # histgram
        if 'show_btot_histogram' in self.p_funcs.keys():

            Btot = si['dfmag_raw']['Btot']
            Dist_au = si['Dist_au_raw']

            print('Interpolating Dist_au...')

            # interpolate Dist_au with scipy.interpolate.interp1d
            ts = np.array([t.timestamp() for t in Dist_au.index])
            f_dist_au = interp1d(ts, Dist_au.values, 'linear', fill_value='extrapolate')

            # create Btot unix index
            Btot_index_unix = np.array([t.timestamp() for t in Btot.index])
            ts = Btot_index_unix

            # print the input value range and Btot index range
            print('Dist_au: %s - %s' %(Dist_au.index[0], Dist_au.index[-1]))
            print('Btot: %s - %s' %(Btot.index[0], Btot.index[-1]))
            print('Btot cadance: %s' %(Btot.index[1]-Btot.index[0]))
            print('Total Points: %d' %(len(Btot)))

            r = f_dist_au(ts)

            # get rid of weird values in Btot
            print('Warning: Btot, %d out of %d is less than 0.1 nT' %(np.sum(Btot < 0.1), len(Btot)))
            print('Warning: Btot, %d out of %d is NaN' %(np.sum(np.isnan(Btot)), len(Btot)))

            self.Btot_temp = Btot
            self.r_temp = r
            Btot.loc[Btot.index[Btot < 0.1]] = np.nan

            # find the rescaling scale with r
            ind = np.invert((np.isnan(r)) | np.isnan(Btot))

            f = lambda x, a, b: a*x+b
            rfit = curve_fit(f, np.log10(r[ind]), np.log10(Btot[ind]))
            scale = -rfit[0][0]

            # normalize btot with r
            Btot1 = Btot * ((r/np.mean(r))**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)

            # mean and std
            bmean0 = np.nanmean(Btot)
            bstd0 = np.nanstd(Btot)
            bmean = np.nanmean(Btot1)
            bstd = np.nanstd(Btot1)
            
            # show normality test
            x = (Btot1.values[np.invert(np.isnan(Btot1.values))])
            if 'nbins' in self.p_funcs['show_btot_histogram'].keys():
                nbins = self.p_funcs['show_btot_histogram']['nbins']
                nstd = self.p_funcs['show_btot_histogram']['nstd']
            else:
                nbins = 201
                nstd = 5

            print('nbins: %d' %(nbins))

            # js distance for Btot
            x = (Btot.values[np.invert(np.isnan(Btot.values))])
            x = (x-np.nanmean(x))/np.nanstd(x)
            bins = np.linspace(-nstd,nstd,nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)
            js_div0 = jensenshannon(hist_data, pdf_gaussian)
            
            # js distance for Btot1
            x = (Btot1.values[np.invert(np.isnan(Btot1.values))])
            x = (x-np.nanmean(x))/np.nanstd(x)
            bins = np.linspace(-nstd,nstd,nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)
            js_div = jensenshannon(hist_data, pdf_gaussian)


            # show the histogram of Btot
            self.fig_btot_hist, self.ax_btot_hist = plt.subplots(1,2,figsize=(11,5), layout='constrained')

            plt.suptitle(r"J-S Distance: $10^{%.4f}$, nstd = %.2f" %(np.log10(js_div), nstd), fontsize = 'x-large')

            # ------------------------------- #
            # left plot (scaled Btot)
            bins_plot = np.linspace(bmean-5*bstd, bmean+5*bstd, 201)
            plt.sca(self.ax_btot_hist[0])
            plt.hist(
                Btot1, bins = bins_plot, histtype = 'step', density = True, color = 'C2'
            )
            plt.hist(
                Btot1, bins = bins_plot, histtype = 'bar', density = True, label = r'$B^{*} = |B| \cdot (r/r0)^{%.4f}$' %(scale), color = 'darkblue', alpha = 0.2
            )
            plt.hist(
                Btot, bins = nbins, histtype = 'step', density = True, label = '|B|', color = 'darkred', ls = '--', alpha = 0.7
            )
            # plt.xlim([-1, np.max(Btot)*1.05])
            plt.axvline(x = bmean, ls = '--', color = 'C0', label = '<$B^{*}$> = %.2f' %(np.mean(Btot1)))
            plt.axvline(x = bmean-bstd, ls = '--', color = 'C1', label = r'$\sigma_{B^{*}}$ = %.2f' %(np.std(Btot1)))
            plt.axvline(x = bmean+bstd, ls = '--', color = 'C1')
            plt.axvspan(
                bmean-3*bstd, bmean+3*bstd,
                alpha = 0.1, color = 'r', label = r'$3\sigma$'
            )
            # over plot gaussian
            x_data = np.linspace(np.nanmean(Btot1)-4*np.nanstd(Btot1), np.nanmean(Btot1)+4*np.nanstd(Btot1), 1000)
            y_data = stats.norm.pdf(x_data, np.nanmean(Btot1), np.nanstd(Btot1))
            plt.plot(
                x_data, y_data, 'k--'
            )
            plt.xlim([bmean-5*bstd, bmean+5*bstd])
            plt.legend(fontsize = 'medium')

            # ------------------------------- #
            # right plot (logged plot)

            plt.sca(self.ax_btot_hist[1])
            plt.hist(
                Btot1, bins = bins_plot, histtype = 'step', density = True, color = 'C2'
            )
            plt.hist(
                Btot1, bins = bins_plot, histtype = 'bar', density = True, label = r'$B^{*} = |B| \cdot (r/r0)^{%.4f}$' %(scale), color = 'darkblue', alpha = 0.2
            )
            plt.hist(
                Btot, bins = nbins, histtype = 'step', density = True, label = '|B|', color = 'darkred', ls = '--', alpha = 0.7
            )
            # plt.xlim([-1, np.max(Btot)*1.05])
            plt.axvline(x = bmean0, ls = '--', color = 'C0', label = '<$B^{*}$> = %.2f' %(np.mean(Btot1)))
            plt.axvline(x = bmean0-bstd0, ls = '--', color = 'C1', label = r'$\sigma_{B^{*}}$ = %.2f' %(np.std(Btot1)))
            plt.axvline(x = bmean0+bstd0, ls = '--', color = 'C1')
            plt.axvspan(
                bmean0-3*bstd0, bmean0+3*bstd0,
                alpha = 0.1, color = 'r', label = r'$3\sigma$'
            )
            # over plot gaussian
            x_data = np.linspace(np.nanmean(Btot1)-4*np.nanstd(Btot1), np.nanmean(Btot1)+4*np.nanstd(Btot1), 1000)
            y_data = stats.norm.pdf(x_data, np.nanmean(Btot1), np.nanstd(Btot1))
            plt.plot(
                x_data, y_data, 'k--'
            )
            plt.semilogy()
            plt.xlim([bmean0-5*bstd0, bmean0+5*bstd0])
            plt.legend(fontsize = 'medium')


        # psd
        if 'PSD' in self.p_funcs.keys():

            self.bpfg_PSD = BPFG(
                si['PSD']['sm_freqs'], si['PSD']['sm_PSD'], label = 'sm_PSD_FFT',
                secondary={
                    'x': si['PSD']['freqs'],
                    'y': si['PSD']['PSD'],
                    'label': 'PSD_FFT'
                },
                diagnostics = si['PSD']['diagnostics']
            )

        # structure func: dB
        if 'struc_funcs' in self.p_funcs.keys():

            self.struc_funcs = si['struc_funcs']
            struc_funcs = si['struc_funcs']

            if self.p_funcs['struc_funcs'] == 1:
                print("Showing dBvecnorms")
                # freq /2 -> freq for FFT
                self.bpfg = BPFG(
                    1/(struc_funcs['dts']/1000*2), struc_funcs['dBvecnorms'], 
                    diagnostics = si['struc_funcs']['diagnostics']
                    )
            elif self.p_funcs['struc_funcs'] == 2:
                print("Showing dBvecs")
                self.bpfg = BPFG(
                    1/(struc_funcs['dts']/1000*2), struc_funcs['dBvecs'], 
                    diagnostics = si['struc_funcs']['diagnostics']
                    )
            elif self.p_funcs['struc_funcs'] == 3:
                self.bpfg = BPFG(
                    1/(struc_funcs['dts']/1000*2), struc_funcs['dBmodnorms'], 
                    diagnostics = si['struc_funcs']['diagnostics']
                    )
            else:
                raise ValueError("Wrong Struc_Func!")

            self.bpfg.connect()
            if np.nanmin(struc_funcs['dBvecnorms']) > 1e-1:
                self.bpfg.arts['PSD']['ax'].set_ylim([1e-1,5e0])
            elif np.nanmin(struc_funcs['dBvecnorms']) > 1e-2:
                self.bpfg.arts['PSD']['ax'].set_ylim([1e-2,1e0])

            
        # wavelet PSD:
        if 'wavelet_PSD' in self.p_funcs.keys():

            self.wavelet_PSD = si['wavelet_PSD']
            freqs_wl, PSD_wl = si['wavelet_PSD']['freqs'], si['wavelet_PSD']['PSD']
            freqs_FFT, PSD_FFT = si['wavelet_PSD']['freqs_FFT'], si['wavelet_PSD']['PSD_FFT']
            sm_freqs_FFT, sm_PSD_FFT = si['wavelet_PSD']['sm_freqs_FFT'], si['wavelet_PSD']['sm_PSD_FFT']

            import_coi = None
            try:
                coi, scales = si['wavelet_PSD']['coi'], si['wavelet_PSD']['scales']

                # show coi range, threshold = 80% under coi
                if 'coi_thresh' in self.p_funcs['wavelet_PSD'].keys():
                    coi_thresh = self.p_funcs['wavelet_PSD']['coi_thresh']
                else:
                    coi_thresh = 0.8
                ind = np.array([np.sum(s < coi)/len(coi) for s in scales]) < coi_thresh

                import_coi = {
                    'range': [freqs_wl[ind][0], freqs_wl[ind][-1]],
                    'label': r'outside COI > %.1f %%' %((1-coi_thresh)*100)
                }

            except:
                print("COI information is not present in wavelet_PSD")

            if "incompressible_only" not in self.p_funcs['wavelet_PSD'].keys():
                self.bpfg_wl = BPFG(
                        freqs_wl, PSD_wl, label = 'PSD_WL',
                        diagnostics = self.wavelet_PSD['diagnostics'],
                        secondary = {
                            'x': freqs_FFT,
                            'y': PSD_FFT,
                            'label': 'PSD_FFT'
                        },
                        third = {
                            'x': sm_freqs_FFT,
                            'y': sm_PSD_FFT,
                            'label': 'sm_PSD_FFT'
                        },
                        import_coi = import_coi
                        )
                self.bpfg_wl.connect()


            else:
                Br, Bt, Bn, Btot = si['dfmag']['Bx'], si['dfmag']['By'], si['dfmag']['Bz'], si['dfmag']['Btot']
                res = si['high_res_infos']['resolution']
                Bmean = np.mean(Btot)
                Bstd = np.std(Btot)
                mode_list = ['2std','1std','10percent']
                try:
                    keep_mode = self.p_funcs['wavelet_PSD']['incompressible_only']['keep_mode']
                    if keep_mode == '2std':
                        keep_ind = (Btot > Bmean - 2*Bstd) & (Btot < Bmean + 2*Bstd)
                    elif keep_mode == '1std':
                        keep_ind = (Btot > Bmean - 1*Bstd) & (Btot < Bmean + 1*Bstd)
                    elif keep_mode == '10percent':
                        keep_ind = (Btot > Bmean*0.9) & (Btot < Bmean*1.1)
                except:
                    # default keep mode 2 std
                    print("Supported modes: "+"".join(["%s, "%(s) for s in mode_list]))
                    print("Using default: keep_mode == 2std")
                    keep_mode = '2std'
                    keep_ind = (Btot > Bmean - 2*Bstd) & (Btot < Bmean + 2*Bstd)

                _,_,_,freqs_wl,PSD_wl,scales,coi = trace_PSD_wavelet(
                    Br.values, Bt.values, Bn.values, res, 
                    dj = 1./12, keep_ind = keep_ind
                )

                si['wavelet_PSD']['incompressible_only'] = {
                    'freqs': freqs_wl,
                    'PSD': PSD_wl,
                    'scales': scales,
                    'coi': coi,
                    'Bmean': Bmean,
                    'Bstd': Bstd,
                    'keep_mode': keep_mode,
                    'keep_ind': keep_ind
                }

                self.bpfg_wl = BPFG(
                        freqs_wl, PSD_wl, label = 'PSD_WL_incompressible_only',
                        diagnostics = self.wavelet_PSD['diagnostics'],
                        secondary = {
                            'x': si['wavelet_PSD']['freqs'],
                            'y': si['wavelet_PSD']['PSD'],
                            'label': 'PSD_WL'
                        },
                        import_coi = import_coi
                        )
                self.bpfg_wl.connect()


        if 'wavelet_rainbow' in self.p_funcs.keys():
            if 'wavelet_rainbows' in si.keys():
                diags = si['wavelet_rainbows']
            else:
                diags = WaveletRainbow(
                    si['dfmag'], si['high_res_infos']['resolution'] 
                )

                si['wavelet_rainbows'] = diags

            N = diags['N']

            fig, axes = plt.subplots(1,2,figsize = [12,6])

            plt.sca(axes[0])

            coi, scales = diags['0']['coi'], diags['0']['scales']
            coi_thresh = 0.4

            ind = np.array([np.sum(s < coi)/len(coi) for s in scales]) < coi_thresh
            f0 = diags['0']['freqs'][ind][0]
            f1 = diags['0']['freqs'][ind][-1]

            plt.axvspan(f0, f1, color = 'r', alpha = 0.05, label = 'coi thresh = %.2f' %(coi_thresh))

            plt.legend(fontsize = 'large')

            colors = plt.cm.rainbow(np.linspace(0, 1, N))
            for i1 in range(N):
                plt.loglog(diags['%d'%(i1)]['freqs'], diags['%d'%(i1)]['PSD'], color = colors[i1], alpha = 0.5)
                
            try:
                freqs_wl, PSD_wl = si['wavelet_PSD']['freqs'], si['wavelet_PSD']['PSD']    
                plt.loglog(freqs_wl, PSD_wl, 'k--')
            except:
                print("WL Rainbow: no wl present...")


            plt.sca(axes[1])

            for i1 in range(N):
                plt.loglog(1/(2*diags['%d'%(i1)]['struc_funcs']['dts']/1000), diags['%d'%(i1)]['struc_funcs']['dBvecnorms'], color = colors[i1])

