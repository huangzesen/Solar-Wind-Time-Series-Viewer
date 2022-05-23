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
T_to_Gauss = 1e4



class TimeSeriesViewer:
    """ View Time Series and Select Intervals """

    def __init__(
        self, paths = None
    ):
        """ Initialize the class """

        print("Initializing...")

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

        # load pickles
        print("Loading PSP pickles...")
        self.dfmag_psp = pd.read_pickle(self.paths['psp_mag_sc'])
        self.dfpar_psp = pd.read_pickle(self.paths['psp_par_sc'])
        self.dfdis_psp = pd.read_pickle(self.paths['psp_dis'])

        print("Loading SolO pickles...")
        self.dfmag_solo = pd.read_pickle(self.paths['solo_mag_sc'])
        self.dfpar_solo = pd.read_pickle(self.paths['solo_par_sc'])
        self.dfdis_solo = pd.read_pickle(self.paths['solo_dis'])

        # collect garbage
        collect()

        print("Done.")


    def ProcessTimeSeries(self):
        """ Process time series to produce desired data product """

        dfts = self.dfts

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

        # Estimate fluctuations of fields #
        va_r = Va_r - np.nanmean(Va_r);   v_r = vr - np.nanmean(vr)
        va_t = Va_t - np.nanmean(Va_t);   v_t = vt - np.nanmean(vt)
        va_n = Va_n - np.nanmean(Va_n);   v_n = vn - np.nanmean(vn)

        # Use moving mean to estimate fluctuation of fields
        va_r = Va_r - Va_r0;   v_r = vr - vr0
        va_t = Va_t - Va_t0;   v_t = vt - vt0
        va_n = Va_n - Va_n0;   v_n = vn - vn0

        Z_plus_squared  = (v_r + va_r)**2 +  (v_t + va_t)**2 + ( v_n + va_n)**2
        Z_minus_squared = (v_r - va_r)**2 +  (v_t - va_t)**2 + ( v_n - va_n)**2
        Z_amplitude     = np.sqrt( (Z_plus_squared + Z_minus_squared)/2 )    

        # cross helicity
        sigma_c         =  (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared)
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
        dfts['di'] = di

        # ion gyro radius
        rho_ci = 10.43968491 * dfts['Vth']/Bmod #in [km]
        dfts['rho_ci'] = rho_ci

        # Plasma Beta
        km2m        = 1e3
        nT2T        = 1e-9
        cm2m        = 1e-2
        B_mag       = dfts['B'] * nT2T                        # |B| units:      [T]
        temp        = 1./2 * m_p * (dfts['Vth']*km2m)**2      # in [J] = [kg] * [m]^2 * [s]^-2
        dens        = dfts['np']/(cm2m**3)                    # number density: [m^-3] 
        beta        = (dens*temp)/((B_mag**2)/(2*mu_0))         # plasma beta
        dfts['beta'] = beta  

        # SW speed
        vsw = np.sqrt(vr**2+vt**2+vn**2)
        dfts['vsw'] = vsw

    
    def FindTimeSeries(self, sc, start_time, end_time, resample_rate = '5min', load_rtn = True):
        """ 
        Find time series in dataframe and create a dfts dataframe 
        start_time and end_time should be either datetime64[ns] or timestamp
        sc: 0-PSP, 1-SolO
        """

        if sc == 0:
            dfmag = self.dfmag_psp
            dfpar = self.dfpar_psp
            dfdis = self.dfdis_psp
        elif sc == 1:
            dfmag = self.dfmag_solo
            dfpar = self.dfpar_solo
            dfdis = self.dfdis_solo
        else:
            raise ValueError("SC = %d not supported!" % (sc))

        indmag = (dfmag.index > start_time) & (dfmag.index < end_time)
        indpar = (dfpar.index > start_time) & (dfpar.index < end_time)
        inddis = (dfdis.index > start_time) & (dfdis.index < end_time)

        # create dfts (dataframe time series)
        dfts = pd.DataFrame(index = dfmag.index[indmag])

        # join time series
        dfts = dfts.join(dfmag[indmag]).join(dfpar[indpar]).join(dfdis[inddis])

        # load rtn magnetic field from SPDF
        if load_rtn:
            pass
        else:
            pass

        # resample the time series
        dfts = dfts.resample(resample_rate).mean()

        # store the time series dataframe
        self.dfts = dfts

        # collect garbage
        collect()


    def LoadSPDF(self, 
        spacecraft = 'PSP',
        start_time = '2021-12-11T23:53:31.000Z', 
        end_time = '2021-12-12T00:53:31.000Z',
        verbose = False
        ):
        """ load data from spdf API """

        # dict to store all the SPDF information
        spdf_data = {
            'spacecraft' : spacecraft,
            'start_time': start_time,
            'end_time': end_time
        }
        
        if verbose:
            print("Spacecraft: ", spacecraft)
            print("Start Time: "+str(self.spdf_data['start_time']))
            print("End Time: "+str(self.spdf_data['end_time']))

        if spacecraft == 'PSP':

            # load ephemeris
            if verbose: print("Loading PSP_HELIO1DAY_POSITION from CDAWEB...")
            vars = ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']
            time = [start_time, end_time]
            status, data = cdas.get_data('PSP_HELIO1DAY_POSITION', vars, time[0], time[1])

            spdf_data['ephem_status'] = status
            spdf_data['ephem_data'] = data

            # load magnetic field
            if verbose: print("Loading PSP_FLD_L2_MAG_RTN_1MIN from CDAWEB...")
            vars = ['psp_fld_l2_mag_RTN_1min']
            time = [start_time, end_time]
            status, data = cdas.get_data('PSP_FLD_L2_MAG_RTN_1MIN', vars, time[0], time[1])

            spdf_data['mag_status'] = status
            spdf_data['mag_data'] = data

        elif spacecraft == 'SolO':

            # load ephemeris
            if verbose: print("Loading SOLO_HELIO1DAY_POSITION from CDAWEB...")
            vars = ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']
            time = [start_time, end_time]
            status, data = cdas.get_data('SOLO_HELIO1DAY_POSITION', vars, time[0], time[1])

            spdf_data['ephem_status'] = status
            spdf_data['ephem_data'] = data

            # load magnetic field
            if verbose: print("Loading SOLO_L2_MAG-RTN-NORMAL-1-MINUTE from CDAWEB...")
            vars = ['B_RTN']
            time = [start_time, end_time]
            status, data = cdas.get_data('SOLO_L2_MAG-RTN-NORMAL-1-MINUTE', vars, time[0], time[1])

            spdf_data['mag_status'] = status
            spdf_data['mag_data'] = data

        else:
            print("option: %s under development...." %(spacecraft))
            pass

        print("Done...")

        return spdf_data




