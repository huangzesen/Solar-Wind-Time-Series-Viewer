# %matplotlib tk
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

from datetime import datetime

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

from TSUtilities import SolarWindCorrelationLength, TracePSD, DrawShadedEventInTimeSeries, smoothing_function
from BreakPointFinderLite import BreakPointFinder
from BreakPointFinderGeneral import BreakPointFinder as BPFG
from LoadData import LoadTimeSeriesWrapper, LoadHighResMagWrapper
from StructureFunctions import MagStrucFunc

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
    ):
        """ Initialize the class """

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

        # calculate the averged field
        keys = ['Br','Bt','Bn','Vr','Vt','Vn','Bx','By','Bz','Vx','Vy','Vz']

        for k in keys:
            if k in dfts_raw0.columns:
                dfts_raw0[k+'0'] = dfts_raw0[k].rolling(self.rolling_rate).mean()
            else:
                print("%s not in columns...!" %(k))

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


    def FindTimeSeries(self, start_time, end_time, verbose = False):
        """ 
        Find time series in dataframe and create a dfts dataframe 
        start_time and end_time should be either datetime64[ns] or timestamp
        sc: 0-PSP, 1-SolO, 2-Helios1, 3-Helios2
        """

        # find the time series between start and end time
        ind = (self.dfts_raw0.index > start_time) & (self.dfts_raw0.index < end_time) 

        # store the time series dataframe
        self.dfts_raw = self.dfts_raw0[ind]

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
        verbose = self.verbose

        """resample the time series"""
        dfts = dfts.resample(resample_rate).mean()


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
        Bmod0 = (dfts['Br0']**2 + dfts['Bt0']**2 + dfts['Bn0']**2).apply('sqrt')

        Va_r = 1e-15* dfts['Br']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_t = 1e-15* dfts['Bt']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_n = 1e-15* dfts['Bn']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]

        Va_r0 = 1e-15* dfts['Br0']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_t0 = 1e-15* dfts['Bt0']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
        Va_n0 = 1e-15* dfts['Bn0']/np.sqrt(mu0*dfts['np']*m_p)   ### Multuply by 1e-15 to get units of [Km/s]

        vr = dfts['Vr']; vt = dfts['Vt']; vn = dfts['Vn']; 
        vr0 = dfts['Vr0']; vt0 = dfts['Vt0']; vn0 = dfts['Vn0']; 

        # # Estimate fluctuations of fields #
        # va_r = Va_r - np.nanmean(Va_r);   v_r = vr - np.nanmean(vr)
        # va_t = Va_t - np.nanmean(Va_t);   v_t = vt - np.nanmean(vt)
        # va_n = Va_n - np.nanmean(Va_n);   v_n = vn - np.nanmean(vn)

        # Use moving mean to estimate fluctuation of fields
        va_r = Va_r - Va_r0;   v_r = vr - vr0
        va_t = Va_t - Va_t0;   v_t = vt - vt0
        va_n = Va_n - Va_n0;   v_n = vn - vn0

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
        vsw = np.sqrt(vr**2+vt**2+vn**2)
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
        tadv = (dfts['Dist_au']/dfts['vsw']).to_numpy() * au_to_km /3600
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

        # alfven speed
        valfven = dfts['B']*nT2T/np.sqrt(dfts['np']*1e6*m_p*mu0)
        dfts['valfven'] = valfven

        # update dfts
        self.dfts = dfts

    
    def AxesInit(self):
        """ Initialize Axes """
        fig, axes = plt.subplots(6,1, figsize = [20,10], layout=self.layout)
        
        self.fig = fig
        self.axes = {
            'mag': axes[0],             # magnetic field
            # 'speed': axes[1],           # Vth and Vsw
            'vsw': axes[1],             # just vsw, no log
            'norm': axes[2],            # normalized quantities: sigma_c, sigma_r
            # 'spec': axes[3],          # spectrum
            'density': axes[3],         # density and di and rho_ci
            #'scale': axes[3],           # di and rho_ci
            'ang': axes[4],             # angles
            # 'alfvenicity': axes[5]      # delta |B| / |delta \vec B|
            'rau': axes[5]              # rau
        }
        self.lines = {
            'mag': None,
            'speed': None,
            'norm': None,
            'scale': None,
            'ang': None,
            'beta': None,
            'alfvenicity': None
        }


    def InitFigure(self, start_time, end_time, 
        no_plot = False, 
        clean_intervals = False, 
        auto_connect=False,
        dfts = None
        ):
        """ make figure """

        # default values
        self.start_time = start_time
        self.end_time = end_time

        # get settings
        verbose = self.verbose

        # Prepare Time Series
        if dfts is None:
            if verbose: print("Preparing Time Series....")
            self.PrepareTimeSeries(
                start_time, end_time
                )
            if verbose: print("Done.")
        else:
            if verbose: print("Importing dfts...")
            self.dfts = dfts

        if no_plot:
            return

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
            parmode = self.dfmisc['parmode']
        except:
            parmode = 'None'

        """make title"""
        fig.suptitle(
            "%s to %s, parmode = %s, rolling_rate = %s" %(str(self.start_time), str(self.end_time), parmode, self.rolling_rate)
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
                        dfts[['Br','Bt','Bn','B_RTN']].plot(ax = ax, legend=False, style=['C0','C1','C2','k'], lw = 0.8)
                        ax.legend(['Br [nT]','Bt','Bn','|B|_RTN'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dfts[['Bx','By','Bz','B_SC']].plot(ax = ax, legend=False, style=['C0','C1','C2','k'], lw = 0.8)
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([-lim, lim])
                lines['mag'] = ax.get_lines()

                collect()
        except:
            print("MAG Plotting have some problems...")
            pass

        """vsw"""
        if not(update):
            try:
                ax = axes['vsw']
                # speeds
                dfts[['V']].plot(ax = ax, legend=False, style=['C2'], lw = 0.8)
                ax.legend(['Vsw[km/s]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                if dfts[['V']].max() < 700:
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                if dfts[['V']].max() < 700:
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([-1.05,1.05])
                lines['norm'] = ax.get_lines()
            except:
                print("sigma_c update failed!")

        """magnetic compressibility"""
        if not(update):
            try:
                ax = axes['norm'].twinx()
                axes['mag_compressibility'] = ax
                if (self.sc == 0) | (self.sc == 1):
                    # for psp and solo, the raw resolution is 5s
                    try:
                        Btot = (self.dfts_raw0[['Br','Bt','Bn']]**2).sum(axis = 1)
                    except:
                        dfmag, infos = LoadHighResMagWrapper(self.sc, self.start_time, self.end_time, credentials = self.credentials)
                        Btot = dfmag['Btot']
                elif (self.sc == 2) | (self.sc == 3) | (self.sc == 4):
                    # for Helios-1/2 and Ulysses, load high-res mag data for this line
                    dfmag, infos = LoadHighResMagWrapper(self.sc, self.start_time, self.end_time, credentials = self.credentials)
                    Btot = dfmag['Btot']
                else:
                    pass
                ind = (Btot.index >= self.start_time) & (Btot.index <= self.end_time)
                Btot = Btot[ind]

                mag_coms = {}
                mckeys = ['1000s','3600s', '10000s']
                for k in mckeys:
                    mag_coms[k] = ((Btot.rolling(k).std())/(Btot.rolling(k).mean())).resample(self.resample_rate).mean()

                self.mag_coms = mag_coms

                for i1 in range(len(mckeys)):
                    k = mckeys[i1]
                    mag_coms[k].plot(ax = ax, style = ['C%d'%(i1+1)], lw = 0.8, alpha = 0.8)

                ax.set_yscale('log')
                ax.set_ylim([10**(-2.05), 10**(0.05)])
                ax.axhline(y = 1e-2, color = 'gray', ls = '--', lw = 0.6)
                ax.axhline(y = 1e-1, color = 'gray', ls = '--', lw = 0.6)
                ax.axhline(y = 1e0, color = 'gray', ls = '--', lw = 0.6)

                ax.legend([r'$10^{-3}$ Hz',r'$1H^{-1}$ Hz',r'$10^{-4}$ Hz'], fontsize='medium', frameon=False, bbox_to_anchor=(1.02,0.8), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                lines['mag_compressibility'] = ax.get_lines()
            except:
                raise ValueError("mag compressibility initialization failed!")
        else:
            try:
                ax = axes['mag_compressibility']
                ls = lines['mag_compressibility']
                if (self.sc == 0) | (self.sc == 1):
                    # for psp and solo, the raw resolution is 5s
                    Btot = (self.dfts_raw0[['Br','Bt','Bn']]**2).sum(axis = 1)
                elif (self.sc == 2) | (self.sc == 3) | (self.sc == 4):
                    # for Helios-1/2 and Ulysses, load high-res mag data for this line
                    dfmag, infos = LoadHighResMagWrapper(self.sc, self.start_time, self.end_time, credentials = self.credentials)
                    Btot = dfmag['Btot']
                else:
                    pass
                ind = (Btot.index >= self.start_time) & (Btot.index <= self.end_time)
                Btot = Btot[ind]

                mag_coms = {}
                mckeys = ['1000s','3600s','10000s']
                for k in mckeys:
                    mag_coms[k] = ((Btot.rolling(k).std())/(Btot.rolling(k).mean())).resample(self.resample_rate).mean()

                self.mag_coms = mag_coms

                for i1 in range(len(mckeys)):
                    k = mckeys[i1]
                    ls[i1].set_data(mag_coms[k].index, mag_coms[k].values)

                ax.axhline(y = 1e-2, color = 'gray', ls = '--', lw = 0.6)
                ax.axhline(y = 1e-1, color = 'gray', ls = '--', lw = 0.6)
                ax.axhline(y = 1e0, color = 'gray', ls = '--', lw = 0.6)

                ax.legend([r'$10^{-3}$ Hz',r'$1H^{-1}$ Hz',r'$10^{-4}$ Hz'], fontsize='medium', frameon=False, bbox_to_anchor=(1.02,0.8), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                lines['mag_compressibility'] = ax.get_lines()

            except:
                print("mag compressibility update failed!")




        """density"""
        if not(update):
            try:
                ax = axes['density']
                # speeds
                dfts[['np']].plot(ax = ax, legend=False, style=['k'], lw = 0.8)
                ax.legend(['np[cm^-3]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.legend(['np[cm^-3]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(dfts['np']),1.05*np.nanmax(dfts['np'])])
                lines['density'] = ax.get_lines()
            except:
                pass

        """scales"""
        if not(update):
            try:
                ax = axes['density'].twinx()
                self.axes['scale'] = ax
                dfts[['di','rho_ci']].plot(ax = ax, legend=False, style=['C3--','C4--'], lw = 0.8, alpha = 0.6)
                ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
                if dfts['di'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                try:
                    min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
                    min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
                    min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                    ax.set_ylim([0.95*min0, 1.05*max0])
                except:
                    if verbose: print("Setting scale limit failed... omitting...")
                lines['scale'] = ax.get_lines()
            except:
                raise ValueError("Initializing scale failed!")
        else:
            try:
                ax = axes['scale']
                ls = lines['scale']
                ls[0].set_data(dfts['di'].index, dfts['di'].values)
                ls[1].set_data(dfts['rho_ci'].index, dfts['rho_ci'].values)
                ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.7), loc = 2)
                if dfts['di'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                try:
                    min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
                    min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
                    min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                    ax.set_ylim([0.95*min0, 1.05*max0])
                except:
                    if verbose: print("Setting scale limit failed... omitting...")
                lines['scale'] = ax.get_lines()
            except:
                raise ValueError("Updating scale failed!...")

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

                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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

                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                (ydata).plot(ax = ax, style = ['k'], legend = False, lw = 0.8)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
        else:
            try:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['Dist_au']*au_to_rsun
                ls[0].set_data(dfts['Dist_au'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                ls[0].set_data(dfts['RAD_AU'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()

        """tadv"""
        if not(update):
            try:
                ax = axes['rau'].twinx()
                axes['tadv'] = ax
                dfts['tadv'].plot(ax = ax, style = ['r'], legend = False, lw = 0.8)
                ax.legend([r'$\tau_{adv}\ [Hr]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
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
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([np.nanmin(dfts['tadv'])*0.95, np.nanmax(dfts['tadv'])*1.05])
                lines['tadv'] = ax.get_lines()
            except:
                pass


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

                i1 = self.FindSelectedIntervals(x)
                selected_interval = self.selected_intervals[i1]
                t0 = selected_interval['start_time']; t1 = selected_interval['end_time']

                # calculate the time series for later diagnostics
                if len(self.p_funcs) > 0:
                    ind = (self.dfts.index > t0) & (self.dfts.index < t1)
                    dftemp = self.dfts.loc[ind]

                    dfmag, infos = LoadHighResMagWrapper(self.sc, t0, t1, credentials = self.credentials)
                    res = infos['resolution']
                    Br = dfmag['Bx'].interpolate().dropna()
                    Bt = dfmag['By'].interpolate().dropna()
                    Bn = dfmag['Bz'].interpolate().dropna()

                    # save the time series to dataframe
                    self.selected_intervals[i1]['TimeSeries'] = dftemp
                    self.selected_intervals[i1]['dfmag'] = dfmag
                    self.selected_intervals[i1]['LTSWsettings'] = self.LTSWsettings

                # psd
                if 'PSD' in self.p_funcs.keys():

                    freq, B_pow = TracePSD(Br.values, Bt.values, Bn.values, res)
                    self.selected_intervals[i1]['PSD'] = {
                        'start_time': t0,
                        'end_time': t1,
                        'sc': self.sc,
                        'freqs': freq,
                        'PSD': B_pow,
                        'resample_info':{
                            'Fraction_missing': dfmag['Bx'].apply(np.isnan).sum()/len(Br)*100,
                            'resolution': res
                        }
                    }
                    # smooth the spectrum
                    if self.calc_smoothed_spec:
                        _, sm_freqs, sm_PSD = smoothing_function(freq, B_pow)
                        self.selected_intervals[i1]['PSD']['sm_freqs'] = sm_freqs
                        self.selected_intervals[i1]['PSD']['sm_PSD'] = sm_PSD

                    if self.useBPF:
                        self.bpf = BreakPointFinder(self.selected_intervals[i1]['PSD'])
                        self.bpf.connect()
                    else:
                        fig1, ax1 = plt.subplots(1, figsize = [6,6])
                        ax1.loglog(freq, B_pow)
                        if self.calc_smoothed_spec:
                            ax1.loglog(sm_freqs, sm_PSD)
                        ax1.set_xlabel(r"$f_{sc}\ [Hz]$", fontsize = 'x-large')
                        ax1.set_ylabel(r"$PSD\ [nT^2\cdot Hz^{-1}]$", fontsize = 'x-large')

                # structure func: dB
                if 'Struc_Func' in self.p_funcs.keys():
                    maxtime = np.log10((t1-t0)/pd.Timedelta('1s')/2)
                    self.struc_funcs = MagStrucFunc(Br, Bt, Bn, (0.5,maxtime), 120)
                    struc_funcs = self.struc_funcs

                    self.selected_intervals[i1]['struc_funcs'] = struc_funcs
                    self.selected_intervals[i1]['struc_funcs']['struc_funcs_diagnostics'] = {}

                    # fig2, ax2 = plt.subplots(1, figsize = [6,6])
                    # ax2.loglog(1/(struc_funcs['dts']/1000), struc_funcs['dBvecnorms'])
                    # # plot 1/3 line
                    # xx = np.logspace(-1,-3)
                    # yy = xx**(-1./3)/10
                    # ax2.loglog(xx, yy)

                    # ax2.set_xlabel(r'1/dts [Hz]')
                    # ax2.set_ylabel(r'dBvecs [nT]')
                    # ax2.set_ylim([1e-1, 4e0])

                    if self.p_funcs['Struc_Func'] == 1:
                        print("Showing dBvecnorms")
                        # freq /2 -> freq for FFT
                        self.bpfg = BPFG(
                            1/(struc_funcs['dts']/1000*2), struc_funcs['dBvecnorms'], 
                            diagnostics = self.selected_intervals[i1]['struc_funcs']['struc_funcs_diagnostics']
                            )
                    elif self.p_funcs['Struc_Func'] == 2:
                        print("Showing dBvecs")
                        self.bpfg = BPFG(
                            1/(struc_funcs['dts']/1000*2), struc_funcs['dBvecs'], 
                            diagnostics = self.selected_intervals[i1]['struc_funcs']['struc_funcs_diagnostics']
                            )
                    elif self.p_funcs['Struc_Func'] == 3:
                        self.bpfg = BPFG(
                            1/(struc_funcs['dts']/1000*2), struc_funcs['dBmodnorms'], 
                            diagnostics = self.selected_intervals[i1]['struc_funcs']['struc_funcs_diagnostics']
                            )
                    else:
                        raise ValueError("Wrong Struc_Func!")

                    self.bpfg.connect()
                    if np.nanmin(struc_funcs['dBvecnorms']) > 1e-1:
                        self.bpfg.arts['PSD']['ax'].set_ylim([1e-1,5e0])
                    elif np.nanmin(struc_funcs['dBvecnorms']) > 1e-2:
                        self.bpfg.arts['PSD']['ax'].set_ylim([1e-2,1e0])

                    
                    # self.selected_intervals[i1]['struc_funcs_diagnostics'] = self.bpfg.diagnostics
                        

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
        if event.button is MouseButton.LEFT:
            """left click to select event"""
            x, y = event.xdata, event.ydata
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
                    selected_interval['rects'][k] = ax.axvspan(l1['timestamp'], l2['timestamp'], alpha = 0.02, color = 'red')

                self.selected_intervals.append(selected_interval)

            self.fig.canvas.draw()
        elif event.button is MouseButton.RIGHT:
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
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(True)
            if need_redraw:
                for _, ax in self.axes.items():
                    ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            self.current_x = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp()
            # update the line positions

            # find rau, tadv and calculate the recommended t interval
            # matplotlib and pandas does not work well together
            # solution: https://stackoverflow.com/questions/54035067/matplotlib-event-xdata-out-of-timeries-range?noredirect=1&lq=1
            xt = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate).to_timestamp()
            self.xt = xt
            # try:
            tind = np.where(np.abs(self.dfts.index - xt) < 2*pd.Timedelta(self.resample_rate))[0][0]
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
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU] / %.2f [Rs], tadv = %.2f [Hr], vsw = %.0f [km/s]" %(xt, rau, rau*au_to_rsun, tadv, vsw)
                    )
            # xt = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate)
            # self.text.set_text("%s" %(str(xt)))
            self.text.set_fontsize('xx-large')
            self.text.set_x(-0.1)
            self.text.set_y(1.05)

            for k, ax in self.axes.items():
                # self.horizontal_lines[k].set_ydata(y)
                self.vertical_lines[k].set_xdata(x)
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


    #-------- Obsolete --------#



