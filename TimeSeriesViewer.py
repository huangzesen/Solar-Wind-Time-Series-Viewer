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

from TSUtilities import SolarWindCorrelationLength, TracePSD


class TimeSeriesViewer:
    """ View Time Series and Select Intervals """

    def __init__(
        self, paths = None, 
        load_spdf = True, 
        start_time_0 = None, 
        end_time_0 = None, 
        sc = 0, 
        preload = True, 
        verbose = True,
        rolling_rate = '1H',
        resample_rate = '5min'
    ):
        """ Initialize the class """

        print("Initializing...")
        print("Current Directory: %s" %(os.getcwd()))
        
        # session info
        self.start_time_0 = start_time_0
        self.end_time_0 = end_time_0
        self.sc = sc

        # dataframe:
        self.dfmag = None
        self.dfpar = None
        self.dfdis = None

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
        self.mag_option = {'norm':1, 'sc':1}
        self.spc_only = True

        if paths is None:
            self.paths = {
            'psp_dis': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfdis_psp_merged.pkl",
            'psp_mag_sc': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfmag_psp_merged.pkl",
            'psp_par_sc': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfpar_psp_merged.pkl",
            'solo_dis': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfdis_solo_merged.pkl",
            'solo_mag_sc': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfmag_solo_merged.pkl",
            'solo_par_sc': "/Volumes/GroupSSD/huangzesen/work/BreakPointFinder/data/dfpar_solo_merged.pkl"
            }
        else:
            self.paths = paths

        # Preload Time Series
        if preload:
            if verbose: print("Preloading Dataframe... This may take some time...")
            self.PreLoadSCDataFrame(rolling_rate = rolling_rate)
            if verbose: print("Done.")

        # collect garbage
        collect()

        print("Done.")


    def PreLoadSCDataFrame(self, paths = None, rolling_rate = '1H'):
        """ 
        Load Spacecraft dataframe 
        Keyword: 
            paths: paths for the magnetic field/particle/distance dataframe
            rolling_rate: default rolling rate is 1H, to redo, use anything other than 1H
        """

        if paths is None:
            paths = self.paths
        else:
            pass

        # look for dataframe locally
        try:
            print("Looking for files in local folder: %s" %(str(Path(os.getcwd()).absolute().joinpath("data"))))
            if self.sc == 0:
                print("Loading PSP pickles...")
                dfmag0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfmag_psp_merged.pkl"))
                dfpar0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfpar_psp_merged.pkl"))
                dfdis0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfdis_psp_merged.pkl"))
            elif self.sc == 1:
                print("Loading SolO pickles...")
                dfmag0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfmag_solo_merged.pkl"))
                dfpar0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfpar_solo_merged.pkl"))
                dfdis0 = pd.read_pickle(Path(os.getcwd()).joinpath("data").joinpath("dfdis_solo_merged.pkl"))
            else:
                raise ValueError("SC=%d not supported!" %(self.sc))
        except:
            if self.sc == 0:
                print("Loading PSP pickles...")
                dfmag0 = pd.read_pickle(self.paths['psp_mag_sc'])
                dfpar0 = pd.read_pickle(self.paths['psp_par_sc'])
                dfdis0 = pd.read_pickle(self.paths['psp_dis'])
            elif self.sc == 1:
                print("Loading SolO pickles...")
                dfmag0 = pd.read_pickle(self.paths['solo_mag_sc'])
                dfpar0 = pd.read_pickle(self.paths['solo_par_sc'])
                dfdis0 = pd.read_pickle(self.paths['solo_dis'])
            else:
                raise ValueError("SC=%d not supported!" %(self.sc))

        if rolling_rate != '1H':
            # redo the rolling average
            print("rolling with window= %s ..." %(rolling_rate))
            print("Magnetic Field...")
            dfmag0['Br0'] = dfmag0['Br'].rolling(rolling_rate).mean()
            dfmag0['Bt0'] = dfmag0['Bt'].rolling(rolling_rate).mean()
            dfmag0['Bn0'] = dfmag0['Bn'].rolling(rolling_rate).mean()
            print("Done. Solar Wind...")
            dfpar0['Vr0'] = dfpar0['Vr'].rolling(rolling_rate).mean()
            dfpar0['Vt0'] = dfpar0['Vt'].rolling(rolling_rate).mean()
            dfpar0['Vn0'] = dfpar0['Vn'].rolling(rolling_rate).mean()

        try: print("Initial Session Range: [%s - %s]" %(self.start_time_0, self.end_time_0))
        except: print("No start_time_0 and end_time_0 defined!")

        if (self.start_time_0 is None) & (self.end_time_0 is None):
            self.start_time_0 = dfmag0.index[0]
            self.end_time_0 = dfmag0.index[-1]
        
        if self.start_time_0 < dfmag0.index[0]:
            self.start_time_0 = dfmag0.index[0]

        if self.end_time_0 > dfmag0.index[-1]:
            self.end_time_0 = dfmag0.index[-1]

        print("Final Session Range: [%s - %s]" %(self.start_time_0, self.end_time_0))

        indmag0 = (dfmag0.index > self.start_time_0) & (dfmag0.index < self.end_time_0)
        indpar0 = (dfpar0.index > self.start_time_0) & (dfpar0.index < self.end_time_0)
        inddis0 = (dfdis0.index > self.start_time_0) & (dfdis0.index < self.end_time_0)

        self.dfmag = dfmag0[indmag0]
        self.dfpar = dfpar0[indpar0]
        self.dfdis = dfdis0[inddis0]


    def AxesInit(self):
        """ Initialize Axes """
        fig, axes = plt.subplots(6,1, figsize = [20,10])
        
        self.fig = fig
        self.axes = {
            'mag': axes[0],             # magnetic field
            'speed': axes[1],           # Vth and Vsw
            'norm': axes[2],            # normalized quantities: sigma_c, sigma_r
            # 'spec': axes[3],          # spectrum
            'scale': axes[3],           # di and rho_ci
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
        verbose = True, resample_rate = '5min', no_plot = False, rolling_rate = '1H'
        ):
        """ make figure """

        # default values
        self.start_time = start_time
        self.end_time = end_time
        self.resample_rate = resample_rate
        self.rolling_rate = rolling_rate
        self.verbose = verbose

        # Preload Time Series
        if (self.dfmag is None) | (self.dfpar is None) | (self.dfdis is None):
            if verbose: print("Preloading Dataframe... This may take some time...")
            self.PreLoadSCDataFrame(rolling_rate = rolling_rate)
            if verbose: print("Done.")

        # Prepare Time Series
        if verbose: print("Preparing Time Series....")
        self.PrepareTimeSeries(
            start_time, end_time, 
            verbose = verbose, 
            resample_rate = resample_rate
            )
        if verbose: print("Done.")

        # Initialize Axes
        self.AxesInit()

        # Plot Time Series
        self.PlotTimeSeries()

        # Create Floating text for crosshair
        self.text = self.axes['mag'].text(0.72, 0.9, '', transform=self.axes['mag'].transAxes, fontsize = 'xx-large')

        # self.horizontal_lines = {}
        self.vertical_lines = {}

        # Show cross hair
        for k, ax in self.axes.items():
            # self.horizontal_lines[k] = ax.axhline(color='k', lw=0.8, ls='--')
            self.vertical_lines[k] = ax.axvline(color='k', lw=0.8, ls='--')

        # redraw
        self.fig.canvas.draw()

        # connect to mpl
        self.connect()


    def ExportSelectedIntervals(self):
        """ 
        Export selected Intervals 
        Return:
            intervals   :   list of dictionaries
            keys        :   list of keys for dict
        """
        intervals = []
        keys = ['spacecraft', 'start_time', 'end_time', 'TimeSeries']
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


    def ExportTimeSeries(self, start_time, end_time, 
        rolling_rate = '1H', verbose = True, resample_rate = '5min'
        ):
        """ 
        A wrapper to export time series 
        start_time/end_time should be pd.Timestamp
        will return a data product, but other data can be accessed via:
        self.dfts | self.dfts_raw | self.dfmag | self.dfpar | self.dfdis
        """
        # check time range
        if (start_time < self.start_time_0) | (end_time > self.end_time_0):
            raise ValueError("%s or %s out of range [%s, %s]" %(start_time, end_time, self.start_time_0, self.end_time_0))

        # Preload dataframe
        if (self.dfmag is None) | (self.dfpar is None) | (self.dfdis is None):
            if verbose: print("Preloading Dataframe... This may take some time...")
            self.PreLoadSCDataFrame(rolling_rate = rolling_rate)
            if verbose: print("Done.")

        # Process dataframe
        if verbose: print("Preparing Time Series....")
        self.PrepareTimeSeries(
            start_time, end_time, 
            verbose = verbose, 
            resample_rate = resample_rate
            )
        if verbose: print("Done.")

        return self.dfts


    def UpdateFigure(
        self, start_time, end_time, 
        verbose=False, resample_rate = None, update_resample_rate_only = False
        ):
        """ Update Figure """

        self.start_time = start_time
        self.end_time = end_time

        if resample_rate is None:
            resample_rate = self.resample_rate
        else:
            self.resample_rate = resample_rate

        # Prepare Time Series
        if verbose: print("Preparing Time Series....")
        self.PrepareTimeSeries(
            start_time, end_time, 
            verbose = verbose, 
            resample_rate = resample_rate, 
            update_resample_rate_only = update_resample_rate_only
            )
        if verbose: print("Done.")

        # Update Time Series
        self.PlotTimeSeries(update=True)

        # redraw
        self.fig.canvas.draw()


    def PrepareTimeSeries(
        self, start_time, end_time, 
        resample_rate = '5min', verbose = False, update_resample_rate_only = False
        ):
        """ One function to prepare time series """
        if update_resample_rate_only:
            pass
        else:
            if verbose: print("Finding Corresponding Time Series...")
            self.FindTimeSeries(start_time, end_time, verbose = verbose)

        if verbose: print("Processing the Time Series...")
        self.ProcessTimeSeries(resample_rate = resample_rate, verbose = verbose)

    
    def PlotTimeSeries(self, verbose = False, update=False):
        """ Plot Time Series """

        dfts = self.dfts
        dfts_raw = self.dfts_raw
        axes = self.axes
        fig = self.fig
        lines = self.lines

        if self.spc_only:
            # for solar probe only
            if self.sc == 0:
                ind = dfts['INSTRUMENT_FLAG'] == 1
                dfts = self.dfts.copy()
                dfts.loc[~ind, 
                ['np','Vth','Vr','Vt','Vn','V','sigma_c','sigma_r','vbangle','di','rho_ci','beta','vsw','tadv']
                ] = np.nan

        """make title"""
        fig.suptitle(
            "%s to %s" %(str(self.start_time), str(self.end_time))
            + "\n"
            + "SpaceCraft: %d, Resample Rate: %s, Window Option: %s (key=%d), MAG Frame: %d, Normed Mag : %d" %(self.sc, self.resample_rate, self.length_list[self.length_key], self.length_key, self.mag_option['sc'], self.mag_option['norm'])
            + "\n"
            + "Real Window Size = %s, Current Session (%s - %s)" %(str(self.end_time-self.start_time), self.start_time_0, self.end_time_0),
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
                        dfts[['Br_RTN','Bt_RTN','Bn_RTN','B_RTN']].plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend(['Br [nT]','Bt','Bn','|B|'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dfts[['Br','Bt','Bn','B']].plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend(['Br_SC [nT]','Bt_SC','Bn_SC','|B|_SC'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                elif self.mag_option['norm'] == 1:
                    if self.mag_option['sc'] == 0:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br_RTN'] = self.dfts['Br_RTN'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt_RTN'] = self.dfts['Bt_RTN'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn_RTN'] = self.dfts['Bn_RTN'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_RTN'] = (dftemp[['Br_RTN','Bt_RTN','Bn_RTN']]**2).sum(axis=1).apply(np.sqrt)
                        dftemp.plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend([r'$Br*a(R)^2\ [nT]$',r'$Bt*a(R)\ a(R)=R/0.1AU$',r'$Bn*a(R)$',r'$|B|$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br'] = self.dfts['Br'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt'] = self.dfts['Bt'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn'] = self.dfts['Bn'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B'] = (dftemp[['Br','Bt','Bn']]**2).sum(axis=1).apply(np.sqrt)
                        dfts[['Br','Bt','Bn','B']].plot(ax = ax, legend=False, style=['C0','C1','C2','k--'], lw = 0.8)
                        ax.legend([r'$Br_{SC}*a(R)^2\ [nT]$',r'$Bt_{SC}*a(R)\ a(R)=R/0.1AU$',r'$Bn_{SC}*a(R)$',r'$|B|_{SC}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B'].max()
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
                        ls[0].set_data(dfts['Br_RTN'].index, dfts['Br_RTN'].values)
                        ls[1].set_data(dfts['Bt_RTN'].index, dfts['Bt_RTN'].values)
                        ls[2].set_data(dfts['Bn_RTN'].index, dfts['Bn_RTN'].values)
                        ls[3].set_data(dfts['B_RTN'].index, dfts['B_RTN'].values)
                        ax.legend(['Br [nT]','Bt','Bn','|B|'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        ls[0].set_data(dfts['Br'].index, dfts['Br'].values)
                        ls[1].set_data(dfts['Bt'].index, dfts['Bt'].values)
                        ls[2].set_data(dfts['Bn'].index, dfts['Bn'].values)
                        ls[3].set_data(dfts['B'].index, dfts['B'].values)
                        ax.legend(['Br_SC [nT]','Bt_SC','Bn_SC','|B|_SC'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dfts['B'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                elif self.mag_option['norm'] == 1:
                    if self.mag_option['sc'] == 0:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br_RTN'] = self.dfts['Br_RTN'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt_RTN'] = self.dfts['Bt_RTN'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn_RTN'] = self.dfts['Bn_RTN'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B_RTN'] = (dftemp[['Br_RTN','Bt_RTN','Bn_RTN']]**2).sum(axis=1).apply(np.sqrt)
                        ls[0].set_data(dftemp['Br_RTN'].index, dftemp['Br_RTN'].values)
                        ls[1].set_data(dftemp['Bt_RTN'].index, dftemp['Bt_RTN'].values)
                        ls[2].set_data(dftemp['Bn_RTN'].index, dftemp['Bn_RTN'].values)
                        ls[3].set_data(dftemp['B_RTN'].index, dftemp['B_RTN'].values)
                        ax.legend([r'$Br*a(R)^2\ [nT]$',r'$Bt*a(R)\ a(R)=R/0.1AU$',r'$Bn*a(R)$',r'$|B|$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B_RTN'].max()
                    elif self.mag_option['sc'] == 1:
                        dftemp = pd.DataFrame(index = self.dfts.index)
                        dftemp['Br'] = self.dfts['Br'].multiply((self.dfts['Dist_au']/0.1)**2, axis = 'index')
                        dftemp['Bt'] = self.dfts['Bt'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['Bn'] = self.dfts['Bn'].multiply((self.dfts['Dist_au']/0.1), axis = 'index')
                        dftemp['B'] = (dftemp[['Br','Bt','Bn']]**2).sum(axis=1).apply(np.sqrt)
                        ls[0].set_data(dftemp['Br'].index, dftemp['Br'].values)
                        ls[1].set_data(dftemp['Bt'].index, dftemp['Bt'].values)
                        ls[2].set_data(dftemp['Bn'].index, dftemp['Bn'].values)
                        ls[3].set_data(dftemp['B'].index, dftemp['B'].values)
                        ax.legend([r'$Br_{SC}*a(R)^2\ [nT]$',r'$Bt_{SC}*a(R)\ a(R)=R/0.1AU$',r'$Bn_{SC}*a(R)$',r'$|B|_{SC}$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                        lim = 1.1*dftemp['B'].max()
                    else:
                        raise ValueError("mag_option['sc']==%d not supported!" %(self.mag_option['sc']))
                else:
                    raise ValueError("mag_option['norm']==%d not supported!" %(self.mag_option['norm']))
                # try:
                #     ax = axes['mag']
                #     ls = lines['mag']
                #     ls[0].set_data(dfts['Br_RTN'].index, dfts['Br_RTN'].values)
                #     ls[1].set_data(dfts['Bt_RTN'].index, dfts['Bt_RTN'].values)
                #     ls[2].set_data(dfts['Bn_RTN'].index, dfts['Bn_RTN'].values)
                #     ls[3].set_data(dfts['B_RTN'].index, dfts['B_RTN'].values)
                #     ax.legend(['Br','Bt','Bn','|B|'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                # except:
                #     ax = axes['mag']
                #     ls = lines['mag']
                    # ls[0].set_data(dfts['Br'].index, dfts['Br'].values)
                    # ls[1].set_data(dfts['Bt'].index, dfts['Bt'].values)
                    # ls[2].set_data(dfts['Bn'].index, dfts['Bn'].values)
                    # ls[3].set_data(dfts['B'].index, dfts['B'].values)
                #     ax.legend(['Br_SC','Bt_SC','Bn_SC','|B|_SC'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
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
                min1 = np.nanmin(dfts['V']); max1 = np.nanmax(dfts['V'])
                min2 = np.nanmin(dfts['Vth']); max2 = np.nanmax(dfts['Vth'])
                min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                ax.set_ylim([0.95*min0, 1.05*max0])
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
                min1 = np.nanmin(dfts['V']); max1 = np.nanmax(dfts['V'])
                min2 = np.nanmin(dfts['Vth']); max2 = np.nanmax(dfts['Vth'])
                min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                ax.set_ylim([0.95*min0, 1.05*max0])
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
                ax.legend([r'$\sigma_c$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([-1.05,1.05])
                lines['norm'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['norm']
                ls = lines['norm']
                ls[0].set_data(dfts['sigma_c'].index, dfts['sigma_c'].values)
                # ls[1].set_data(dfts['sigma_r'].index, dfts['sigma_r'].values)
                # ax.legend([r'$\sigma_c$',r'$\sigma_r$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                ax.legend([r'$\sigma_c$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([-1.05,1.05])
                lines['norm'] = ax.get_lines()
            except:
                pass

        """scales"""
        if not(update):
            try:
                ax = axes['scale']
                dfts[['di','rho_ci']].plot(ax = ax, legend=False, style=['C3--','C4--'], lw = 0.8)
                ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                if dfts['di'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
                min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
                min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                ax.set_ylim([0.95*min0, 1.05*max0])
                lines['scale'] = ax.get_lines()
            except:
                pass
        else:
            try:
                ax = axes['scale']
                ls = lines['scale']
                ls[0].set_data(dfts['di'].index, dfts['di'].values)
                ls[1].set_data(dfts['rho_ci'].index, dfts['rho_ci'].values)
                ax.legend([r'$d_i$[km]',r'$\rho_{ci}$[km]'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
                if dfts['di'].apply(np.isnan).sum() != len(dfts):
                    ax.set_yscale('log')
                ax.set_xticks([], minor=True)
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                min1 = np.nanmin(dfts['di']); max1 = np.nanmax(dfts['di'])
                min2 = np.nanmin(dfts['rho_ci']); max2 = np.nanmax(dfts['rho_ci'])
                min0 = np.nanmin([min1,min2]); max0 = np.nanmax([max1, max2])
                ax.set_ylim([0.95*min0, 1.05*max0])
                lines['scale'] = ax.get_lines()
            except:
                pass

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
                    ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    dfts[['vbangle']].plot(ax = ax, legend=False, style = ['r--','b--'], lw = 0.8)
                    ax.legend([r'$<\vec V, \vec B>$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
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
                    ax.legend([r'$<\vec V, \vec B>$',r'$<\vec r, \vec B>$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
                except:
                    ls[0].set_data(dfts['vbangle'].index, dfts['vbangle'].values)
                    ax.legend([r'$<\vec V, \vec B>$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02,1), loc = 2)
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
                ax.legend([r'$\beta$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
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
                ax.axhline(0, color = 'gray', ls = '--', lw = 0.6)
            except:
                pass
        else:
            try:
                ax = axes['beta']
                ls = lines['beta']
                ls[0].set_data(dfts['beta'].index, dfts['beta'].values)
                ax.legend([r'$\beta$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
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
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                (ydata).plot(ax = ax, style = ['k'], legend = False, lw = 0.8)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
        else:
            try:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['Dist_au']*au_to_rsun
                ls[0].set_data(dfts['Dist_au'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()
            except:
                ax = axes['rau']
                ls = lines['rau']
                ydata = dfts['RAD_AU']*au_to_rsun
                ls[0].set_data(dfts['RAD_AU'].index, (ydata).values)
                ax.legend([r'$R\ [R_{\circ}]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 1), loc = 2)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([0.95*np.nanmin(ydata), 1.05*np.nanmax(ydata)])
                lines['rau'] = ax.get_lines()

        """tadv"""
        if not(update):
            try:
                ax = axes['rau'].twinx()
                axes['tadv'] = ax
                dfts['tadv'].plot(ax = ax, style = ['r'], legend = False, lw = 0.8)
                ax.legend([r'$\tau_{adv}\ [Hr]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
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
                ax.legend([r'$\tau_{adv}\ [Hr]$'], fontsize='x-large', frameon=False, bbox_to_anchor=(1.02, 0), loc = 3)
                ax.set_xlim([dfts.index[0], dfts.index[-1]])
                ax.set_ylim([np.nanmin(dfts['tadv'])*0.95, np.nanmax(dfts['tadv'])*1.05])
                lines['tadv'] = ax.get_lines()
            except:
                pass


        # fig.tight_layout()
        self.lines = lines
        self.axes = axes
        collect()


    def ProcessTimeSeries(self, verbose = False, resample_rate = '5min'):
        """ Process time series to produce desired data product """

        load_spdf = self.load_spdf
        dfts = self.dfts_raw.copy()
        # resample_rate = self.resample_rate

        """resample the time series"""
        dfts = dfts.resample(resample_rate).mean()

        """join the spdf data"""
        if load_spdf:
            try:
                dfspdf = self.spdf_data['dfspdf']
                dfts = dfts.join(dfspdf.resample(resample_rate).mean())
                dfts['B_RTN'] = (dfts[['Br_RTN','Bt_RTN','Bn_RTN']]**2).sum(axis = 1).apply(np.sqrt)
                # B R angle (need spdf)
                dfts['brangle'] = (dfts['Br_RTN']/dfts['B_RTN']).apply(np.arccos) * 180 / np.pi
            except:
                pass

        """Modulus of vectors"""
        dfts['B'] = (dfts['Br']**2+dfts['Bt']**2+dfts['Bn']**2).apply(np.sqrt)
        dfts['V'] = (dfts['Vr']**2+dfts['Vt']**2+dfts['Vn']**2).apply(np.sqrt)

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
        B_mag       = dfts['B'] * nT2T                        # |B| units:      [T]
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

        # distance
        try:
            dist = dfts['Dist_au'].to_numpy()
            dist_spdf = dfts['RAD_AU'].to_numpy()
            dist[np.isnan(dist)] = dist_spdf[np.isnan(dist)]
            dfts['Dist_au'] = dist
        except:
            pass

        # advection time [Hr]
        tadv = (dfts['Dist_au']/dfts['vsw']).to_numpy() * au_to_km /3600
        tadv[tadv < 0] = np.nan
        dfts['tadv'] = tadv
        try:
            tadv_spdf = (dfts['RAD_AU']/dfts['vsw']).to_numpy() * au_to_km /3600
            tadv_spdf[tadv_spdf < 0] = np.nan
            dfts['tadv_spdf'] = tadv_spdf
            tadv[np.isnan(tadv)] = tadv_spdf[np.isnan(tadv)]
            dfts['tadv'] = tadv
        except:
            pass

        # update dfts
        self.dfts = dfts


    def FindTimeSeries(self, start_time, end_time, verbose = False):
        """ 
        Find time series in dataframe and create a dfts dataframe 
        start_time and end_time should be either datetime64[ns] or timestamp
        sc: 0-PSP, 1-SolO
        """

        load_spdf = self.load_spdf

        # if sc == 0:
        #     dfmag = self.dfmag_psp
        #     dfpar = self.dfpar_psp
        #     dfdis = self.dfdis_psp
        # elif sc == 1:
        #     dfmag = self.dfmag_solo
        #     dfpar = self.dfpar_solo
        #     dfdis = self.dfdis_solo
        # else:
        #     raise ValueError("SC = %d not supported!" % (sc))

        dfmag = self.dfmag
        dfpar = self.dfpar
        dfdis = self.dfdis

        indmag = (dfmag.index > start_time) & (dfmag.index < end_time)
        indpar = (dfpar.index > start_time) & (dfpar.index < end_time)
        inddis = (dfdis.index > start_time) & (dfdis.index < end_time)

        # create dfts (dataframe time series)
        dfts = pd.DataFrame(index = dfmag.index[indmag])

        if len(dfts) < 5:
            raise ValueError("Not Enough Data Points! %s - %s" %(start_time, end_time))

        # join time series
        dfts = dfts.join(dfmag[indmag]).join(dfpar[indpar]).join(dfdis[inddis])

        # load magnetic field and ephem from SPDF
        if load_spdf:
            # spdf system is buggy, has a constant 4H shift
            t0 = start_time - pd.Timedelta('1d')
            t1 = end_time + pd.Timedelta('1d')
            if self.sc==0:
                self.spdf_data = self.LoadSPDF(spacecraft = 'PSP', start_time = t0, end_time = t1, verbose = verbose)
            elif self.sc==1:
                self.spdf_data = self.LoadSPDF(spacecraft = 'SolO', start_time = t0, end_time = t1, verbose = verbose)
        else:
            pass

        # store the time series dataframe
        self.dfts_raw = dfts

        # collect garbage
        collect()


    def LoadSPDF(self, 
        spacecraft = 'PSP',
        start_time = '2021-12-11T23:53:31.000Z', 
        end_time = '2021-12-12T00:53:31.000Z',
        verbose = False
        ):
        """ load data from spdf API """

        try:

            # convert datetime type
            start_time = np.datetime64(start_time).astype(datetime)
            end_time = np.datetime64(end_time).astype(datetime)

            # dict to store all the SPDF information
            spdf_data = {
                'spacecraft' : spacecraft,
                'start_time': start_time,
                'end_time': end_time
            }
            
            if verbose:
                print("Spacecraft: ", spacecraft)
                print("Start Time: "+str(spdf_data['start_time']))
                print("End Time: "+str(spdf_data['end_time']))

            if spacecraft == 'PSP':

                # load ephemeris
                if verbose: print("Loading PSP_HELIO1DAY_POSITION from CDAWEB...")
                vars = ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']
                time = [start_time, end_time]
                status, data = cdas.get_data('PSP_HELIO1DAY_POSITION', vars, time[0], time[1])

                spdf_data['ephem_status'] = status
                spdf_data['ephem_data'] = data

                dfdis = pd.DataFrame(spdf_data['ephem_data']).set_index("Epoch")
                dfdis = dfdis.resample('1min').mean().interpolate('linear')

                # load magnetic field
                if verbose: print("Loading PSP_FLD_L2_MAG_RTN_1MIN from CDAWEB...")
                vars = ['psp_fld_l2_mag_RTN_1min']
                time = [start_time, end_time]
                status, data = cdas.get_data('PSP_FLD_L2_MAG_RTN_1MIN', vars, time[0], time[1])

                spdf_data['mag_status'] = status
                spdf_data['mag_data'] = data

                dfmag = pd.DataFrame(
                    index = spdf_data['mag_data']['epoch_mag_RTN_1min'],
                    data = spdf_data['mag_data']['psp_fld_l2_mag_RTN_1min'],
                    columns = ['Br_RTN','Bt_RTN','Bn_RTN']
                ).resample('1min').mean()

                spdf_data['dfspdf'] = dfmag.join(dfdis)

            elif spacecraft == 'SolO':

                # load ephemeris
                if verbose: print("Loading SOLO_HELIO1DAY_POSITION from CDAWEB...")
                vars = ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']
                time = [start_time, end_time]
                status, data = cdas.get_data('SOLO_HELIO1DAY_POSITION', vars, time[0], time[1])

                spdf_data['ephem_status'] = status
                spdf_data['ephem_data'] = data

                dfdis = pd.DataFrame(spdf_data['ephem_data']).set_index("Epoch")
                dfdis = dfdis.resample('1min').mean().interpolate('linear')

                # load magnetic field
                if verbose: print("Loading SOLO_L2_MAG-RTN-NORMAL-1-MINUTE from CDAWEB...")
                vars = ['B_RTN']
                time = [start_time, end_time]
                status, data = cdas.get_data('SOLO_L2_MAG-RTN-NORMAL-1-MINUTE', vars, time[0], time[1])

                spdf_data['mag_status'] = status
                spdf_data['mag_data'] = data

                dfmag = pd.DataFrame(
                    index = spdf_data['mag_data']['EPOCH'],
                    data = spdf_data['mag_data']['B_RTN'],
                    columns = ['Br_RTN','Bt_RTN','Bn_RTN']
                ).resample('1min').mean()

                spdf_data['dfspdf'] = dfmag.join(dfdis)

            else:
                raise ValueError("spacecraft = %s not supported!" %(spacecraft))

            if verbose: print("Done...")

            return spdf_data
        except:
            return None


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
            self.start_time, self.end_time, verbose = self.verbose
        )


    #--------  Visual Interface --------#

    def on_key_event(self, event):
        """ On Key Event """

        # ------ character key for functions ------ #
        if (event.key == 'd'):
            try:
                x = self.current_x
                for i1 in range(len(self.selected_intervals)):
                    selected_interval = self.selected_intervals[i1]
                    if (x > selected_interval['start_time']) & (x < selected_interval['end_time']):
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
                    self.start_time, self.end_time, verbose=self.verbose, resample_rate = self.resample_rate
                )
            except:
                pass
        elif (event.key == 'p'):
            try:
                x = self.current_x
                for i1 in range(len(self.selected_intervals)):
                    selected_interval = self.selected_intervals[i1]
                    if (x > selected_interval['start_time']) & (x < selected_interval['end_time']):
                        t0 = selected_interval['start_time']; t1 = selected_interval['end_time']
                        ind = (self.dfts_raw.index > t0) & (self.dfts_raw.index < t1)
                        dftemp = self.dfts_raw.loc[ind,['Br','Bt','Bn','Vr','Vt','Vn','np','Vth']]
                        Br = dftemp['Br'].values
                        Bt = dftemp['Bt'].values
                        Bn = dftemp['Bn'].values
                        freq, B_pow = TracePSD(Br, Bt, Bn, 1)
                        fig1, ax1 = plt.subplots(1, figsize = [6,6])
                        ax1.loglog(freq, B_pow)
                        ax1.xlabel(r"$f_{sc}\ [Hz]$", fontsize = 'x-large')
                        ax1.ylabel(r"$PSD\ [nT^2\cdot Hz^{-1}]$", fontsize = 'x-large')
                        # save the time series to dataframe
                        self.selected_intervals[i1]['TimeSeries'] = dftemp
            except:
                pass
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
                self.start_time, self.end_time, verbose=self.verbose
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
                self.start_time, self.end_time, verbose=self.verbose
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
                self.start_time, self.end_time, verbose=self.verbose,
                resample_rate = self.resample_rate, 
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
                self.start_time, self.end_time, verbose=self.verbose,
                resample_rate = self.resample_rate, 
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
                self.start_time, self.end_time, verbose=self.verbose
                )
                for k, ax in self.axes.items():
                    l1['lines'][k].remove()
                    l2['lines'][k].remove()
                    ax.figure.canvas.draw()
                
                # empty the stack
                self.green_vlines = []

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
                rau = self.dfts['RAD_AU'][tind]

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
                        "%s, R[AU] = %.2f [AU], tadv = %.2f [Hr], Selected = %.2f [Hr], Recommended = %.2f [Hr], Corrected with tadv = %.2f [Hr]" %(xt, rau, tadv, dtrl, t_len1, t_len2)
                    )
                except:
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU], tadv = %.2f [Hr], Recommended = %.2f [Hr], Corrected with tadv = %.2f [Hr]" %(xt, rau, tadv, t_len1, t_len2)
                    )
            else:
                try:
                    x = self.current_x
                    for i1 in range(len(self.selected_intervals)):
                        selected_interval = self.selected_intervals[i1]
                        if (x > selected_interval['start_time']) & (x < selected_interval['end_time']):
                            dtint = (selected_interval['end_time']- selected_interval['start_time']).total_seconds()/3600
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU], tadv = %.2f [Hr], Current = %.2f [Hr], Recommended = %.2f [Hr], Corrected with tadv = %.2f [Hr]" %(xt, rau, tadv, dtint, t_len1, t_len2)
                    )
                except:
                    self.text.set_text(
                        "%s, R[AU] = %.2f [AU], tadv = %.2f [Hr], Recommended = %.2f [Hr], Corrected with tadv = %.2f [Hr]" %(xt, rau, tadv, t_len1, t_len2)
                    )
            # xt = pd.Period(ordinal=int(event.xdata), freq=self.resample_rate)
            # self.text.set_text("%s" %(str(xt)))
            self.text.set_fontsize('xx-large')
            self.text.set_x(0.05)
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

