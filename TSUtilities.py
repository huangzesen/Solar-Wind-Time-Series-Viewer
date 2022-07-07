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

import datetime
from numba import jit,njit,prange

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


# -----------  Load Data ----------- #

def LoadTimeSeriesSPDF(sc=None, start_time=None, keys = None, end_time=None, verbose=True, get_keys = False):
    """ Wrapper for loading time series from SPDF """

    # Just return keys
    if (sc is not None) & ((start_time is None) | (end_time is None)) & (get_keys == True):
        t0 = pd.Timestamp("2020-01-01").to_pydatetime()
        t1 = pd.Timestamp("2020-01-01").to_pydatetime()
        if sc == 0:
            return LoadPSPTimeSeriesSPDF(t0, t1, get_keys = True)
        elif sc == 1:
            return LoadSOlOTimeSeriesSPDF(t0, t1, get_keys = True)
        else:
            raise ValueError("sc = %d not supported!" %(sc))
    # load data from spdf
    elif (sc is not None) & (start_time is not None) & (end_time is not None):
        # load data
        if sc == 0:
            return LoadPSPTimeSeriesSPDF(start_time, end_time, get_keys = False, verbose = verbose, keys = keys)
        elif sc == 1:
            return LoadSOlOTimeSeriesSPDF(start_time, end_time, get_keys = False, verbose = verbose, keys = keys)
        else:
            raise ValueError("sc = %d not supported!" %(sc))
    else:
        raise ValueError("Input Error!")


def LoadPSPTimeSeriesSPDF(start_time, end_time, keys = None, verbose = True, get_keys = False):
    """
    Load Parker Solar Probe Time Series, this is a wrapper for SPDF API
    All Time Series will be stored with dataframe, NOT resampled
    Keys available for loading:
    Key:                SPDF Key:                       Description:                                Columns:     
    ephem               PSP_HELIO1DAY_POSITION          Ephemeris                                   ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']                                               
    mag_rtn             PSP_FLD_L2_MAG_RTN              Full resolution magnetic field              ['Br','Bt','Bn']                                        
    mag_sc              PSP_FLD_L2_MAG_SC               Full resolution sc frame magnetic field     ['Bx','By','Bz']
    mag_1min_rtn        PSP_FLD_L2_MAG_RTN_1MIN         1min RTN magnetic field                     ['Br','Bt','Bn']
    mag_1min_sc         PSP_FLD_L2_MAG_SC_1MIN          1min SC magnetic field                      ['Bx','By','Bz']
    mag_scm             PSP_FLD_L3_MERGED_SCAM_WF       Merged FIELDS and SCAM data                 *Keep as dict
    spc                 PSP_SWP_SPC_L3I                 SPC Moments                                 ['Vx','Vy','Vz','np','Vth','Vr','Vt','Vn']   
    span                PSP_SWP_SPI_SF00_L3_MOM_INST    SPAN Moments                                ['Vx','Vy','Vz','np','Vth','TEMP']

    Input:
        start_time  [datetime.datetime]   :   start of the interval
        end_time    [datetime.datetime]   :   end of the interval

    Keywords:
        keys = None             [List]                  can be a list containing all the desired keys
        verbose = True          [Boolean]               more blabla?
        get_keys = False        [Boolean]               just return the spdf_infos dict                    

    Output:
        spdf_data   [dict]                :     a dictionary of everything

        spdf_data:
        Keys            Description                 Datatype
        spacecraft      spacecraft name             string
        sc              spacecraft code             int
        start_time      start of interval           datetime.datetime
        end_time        end of interval             datetime.datetime
        spdf            dictionary of everything    dict

    Note spdf is a dict containing all keys and key_status, e.g. mag_rtn and mag_rtn_status
    """
    # check input type and large & small
    if isinstance(start_time, datetime.datetime) & isinstance(end_time, datetime.datetime):
        if end_time < start_time:
            if verbose: print("Exchanged start and end...")
            end_time, start_time = start_time, end_time
        print("Time range = [%s, %s]" % (str(start_time), str(end_time)))
        if verbose: 
            print("Interval Length = %s" % (str(end_time - start_time)))
        else: pass
    else:
        raise TypeError(
            "start_time and end_time must be datetime.datetime, current type: %s and %s" %(str(type(start_time)), str(type(end_time)))
            )

    # set keys
    allowed_keys0 = ['ephem', 'mag_rtn', 'mag_sc', 'mag_1min_rtn', 'mag_1min_sc', 'mag_scm', 'spc', 'span']
    keys0 = ['ephem', 'mag_rtn', 'mag_sc', 'mag_1min_rtn', 'mag_1min_sc', 'spc', 'span']
    if keys is None:
        keys = keys0
    else:
        for key in keys:
            if key in allowed_keys0: pass
            else: raise ValueError(
                "key: ''%s'' not supported! Supported keys: \n" %(key) 
                + 
                "''%s'' \n"*len(keys0) %(tuple(keys0))
            )

    # dict for SPDF keys
    spdf_infos = {
        'ephem': {
            'key': 'PSP_HELIO1DAY_POSITION',
            'vars': ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_rtn': {
            'key': 'PSP_FLD_L2_MAG_RTN',
            'vars': ['psp_fld_l2_mag_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_sc': {
            'key': 'PSP_FLD_L2_MAG_SC',
            'vars': ['psp_fld_l2_mag_SC'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_1min_rtn': {
            'key': 'PSP_FLD_L2_MAG_RTN_1MIN',
            'vars': ['psp_fld_l2_mag_RTN_1min'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_1min_sc': {
            'key':'PSP_FLD_L2_MAG_SC_1MIN',
            'vars': ['psp_fld_l2_mag_SC_1min'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_scm': {
            'key': 'PSP_FLD_L3_MERGED_SCAM_WF',
            'vars': ['psp_fld_l3_merged_scam_wf_uvw','psp_fld_l3_merged_scam_wf_SC','psp_fld_l3_merged_scam_wf_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
            },
        'spc': {
            'key': 'PSP_SWP_SPC_L3I',
            'vars': ['general_flag','vp_moment_SC_gd','vp_moment_SC_deltahigh_gd','vp_moment_SC_deltalow_gd','vp_moment_RTN_gd','vp_moment_RTN_deltahigh_gd','vp_moment_RTN_deltalow_gd','np_moment_gd','np_moment_deltahigh_gd','np_moment_deltalow_gd','wp_moment_gd','wp_moment_deltahigh_gd','wp_moment_deltalow_gd','vp1_fit_SC_gd','vp1_fit_SC_uncertainty_gd','vp1_fit_RTN_gd','vp1_fit_RTN_uncertainty_gd','np1_fit_gd','np1_fit_uncertainty_gd','wp1_fit_gd','wp1_fit_uncertainty_gd','vp_fit_SC_gd','vp_fit_SC_uncertainty_gd','vp_fit_RTN_gd','vp_fit_RTN_uncertainty_gd','np_fit_gd','np_fit_uncertainty_gd','wp_fit_gd','wp_fit_uncertainty_gd','va_fit_SC_gd','va_fit_SC_uncertainty_gd','va_fit_RTN_gd','va_fit_RTN_uncertainty_gd','na_fit_gd','na_fit_uncertainty_gd','wa_fit_gd','wa_fit_uncertainty_gd','v3_fit_SC_gd','v3_fit_SC_uncertainty_gd','v3_fit_RTN_gd','v3_fit_RTN_uncertainty_gd','n3_fit_gd','n3_fit_uncertainty_gd','w3_fit_gd','w3_fit_uncertainty_gd','sc_pos_HCI','sc_vel_HCI','carr_latitude','carr_longitude','vp_moment_SC','vp_moment_SC_deltahigh','vp_moment_SC_deltalow','vp_moment_RTN','vp_moment_RTN_deltahigh','vp_moment_RTN_deltalow','np_moment','np_moment_deltahigh','np_moment_deltalow','wp_moment','wp_moment_deltahigh','wp_moment_deltalow','vp1_fit_SC','vp1_fit_SC_uncertainty','vp1_fit_RTN','vp1_fit_RTN_uncertainty','np1_fit','np1_fit_uncertainty','wp1_fit','wp1_fit_uncertainty','vp_fit_SC','vp_fit_SC_uncertainty','vp_fit_RTN','vp_fit_RTN_uncertainty','np_fit','np_fit_uncertainty','wp_fit','wp_fit_uncertainty','va_fit_SC','va_fit_SC_uncertainty','va_fit_RTN','va_fit_RTN_uncertainty','na_fit','na_fit_uncertainty','wa_fit','wa_fit_uncertainty','v3_fit_SC','v3_fit_SC_uncertainty','v3_fit_RTN','v3_fit_RTN_uncertainty','n3_fit','n3_fit_uncertainty','w3_fit','w3_fit_uncertainty'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'span': {
            'key': 'PSP_SWP_SPI_SF00_L3_MOM_INST',
            'vars': ['QUALITY_FLAG','DENS','VEL','T_TENSOR','TEMP','MAGF_SC','MAGF_INST','EFLUX_VS_ENERGY','EFLUX_VS_THETA','EFLUX_VS_PHI'],
            'status': None,
            'data': None,
            'dataframe': None
        }
    }

    if get_keys:
        return spdf_infos

    # ------ initialize dictionary ------ #
    spdf_data = {
        'spacecraft'    :   'PSP',
        'sc'            :   0,
        'start_time'    :   start_time,
        'end_time'      :   end_time,
        'spdf_infos'    :   {}
        }
    
    # ------ download data ------ #
    time = [start_time, end_time]
    for key in keys:
        if verbose: print("Loading %s from CDAWEB..." %(spdf_infos[key]['key']))
        status, data = cdas.get_data(spdf_infos[key]['key'], spdf_infos[key]['vars'], time[0], time[1])
        spdf_info = spdf_infos[key]
        spdf_info['status'] = status
        spdf_info['data'] = data
        spdf_data['spdf_infos'][key] = spdf_info

    if verbose: print("Done. Processing Data...")

    # ------ create dataframe ------ #
    # ephem
    try:
        key = 'ephem'
        if key in keys:
            df = pd.DataFrame(spdf_data['spdf_infos'][key]['data']).set_index("Epoch")
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_rtn
    try:
        key = 'mag_rtn'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['epoch_mag_RTN'],
                    data = spdf_data['spdf_infos'][key]['data']['psp_fld_l2_mag_RTN'],
                    columns = ['Br','Bt','Bn']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_sc
    try:
        key = 'mag_sc'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['epoch_mag_SC'],
                    data = spdf_data['spdf_infos'][key]['data']['psp_fld_l2_mag_SC'],
                    columns = ['Bx','By','Bz']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))    

    # mag_1min_rtn
    try:
        key = 'mag_1min_rtn'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['epoch_mag_RTN_1min'],
                    data = spdf_data['spdf_infos'][key]['data']['psp_fld_l2_mag_RTN_1min'],
                    columns = ['Br','Bt','Bn']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_1min_sc
    try:
        key = 'mag_1min_sc'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['epoch_mag_SC_1min'],
                    data = spdf_data['spdf_infos'][key]['data']['psp_fld_l2_mag_SC_1min'],
                    columns = ['Bx','By','Bz']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_scm
    pass

    # spc
    try:
        key = 'spc'
        if key in keys:
            df = pd.DataFrame(index = spdf_data['spdf_infos']['spc']['data']['Epoch'])
            df = df.join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['spc']['data']['vp_moment_SC_gd'],
                    columns = ['Vx','Vy','Vz']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['spc']['data']['vp_moment_RTN_gd'],
                    columns = ['Vr','Vt','Vn']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['spc']['data']['np_moment_gd'],
                    columns = ['np']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['spc']['data']['wp_moment_gd'],
                    columns = ['Vth']
                )
            )
            # pre-clean the data
            indnan = spdf_data['spdf_infos']['spc']['data']['general_flag']!=0
            df.loc[indnan,:] = np.nan
            # more cleaning
            df[df.abs() > 1e20] = np.nan
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))


    # span
    try:
        key = 'span'
        if key in keys:
            df = pd.DataFrame(index = spdf_data['spdf_infos']['span']['data']['Epoch'])
            df = df.join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['DENS'],
                    columns = ['np']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['VEL'],
                    columns = ['Vx','Vy','Vz']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['TEMP'],
                    columns = ['TEMP']
                ))
            # pre-clean the data
            # indnan = spdf_data['spdf_infos']['span']['data']['QUALITY_FLAG'] == int('0000000000000000', 2)
            # df.loc[indnan,:] = np.nan
            # get rid of minus temperature
            df.loc[df['TEMP'] < 0, :] = np.nan
            # get rid of 0.0 in temperature
            df.loc[df['TEMP'].abs() < 1e-2, :] = np.nan
            # calculate Vth
            df['Vth'] = 13.84112218 * np.sqrt(df['TEMP']) # 1eV = 13.84112218 km/s (kB*T = 1/2*mp*Vth^2)
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))



    if verbose: print("Done.")

    
    return spdf_data


def LoadSOlOTimeSeriesSPDF(start_time, end_time, keys = None, verbose = True, get_keys = False):
    """
    Load Solar Orbiter Time Series, this is a wrapper for SPDF API
    All Time Series will be stored with dataframe, NOT resampled
    Keys available for loading:
    Key:                SPDF Key:                           Description:                        Columns:     
    ephem               SOLO_HELIO1DAY_POSITION             Ephemeris                           ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']                                               
    mag_rtn             SOLO_L2_MAG-RTN-NORMAL              Normal resolution MAG RTN           ['Br','Bt','Bn']                                        
    mag_1min_rtn        SOLO_L2_MAG-RTN-NORMAL-1-MINUTE     1min RTN magnetic field             ['Br','Bt','Bn']
    mag_burst_rtn       SOLO_L2_MAG-RTN-BURST               Burst RTN magnetic field            ['Br','Bt','Bn']
    mag_srf             SOLO_L2_MAG-SRF-NORMAL              Normal SRF magnetic field           ['Bx','By','Bz']
    mag_burst_srf       SOLO_L2_MAG-SRF-BURST               Burst SRF magnetic field            ['Bx','By','Bz']
    swa                 SOLO_L2_SWA-PAS-GRND-MOM            SWA Moments                         ['Vx','Vy','Vz','np','Vth','TEMP','Vr','Vt','Vn']   
    
    Input:
        start_time  [datetime.datetime]   :   start of the interval
        end_time    [datetime.datetime]   :   end of the interval

    Keywords:
        keys = None             [List]                  can be a list containing all the desired keys
        verbose = True          [Boolean]               more blabla?
        get_keys = False        [Boolean]               just return the spdf_infos dict     

    Output:
        spdf_data   [dict]                :     a dictionary of everything

        spdf_data:
        Keys            Description                 Datatype
        spacecraft      spacecraft name             string
        sc              spacecraft code             int
        start_time      start of interval           datetime.datetime
        end_time        end of interval             datetime.datetime
        spdf            dictionary of everything    dict

    Note spdf is a dict containing all keys and key_status, e.g. mag_rtn and mag_rtn_status
    """
    # check input type and large & small
    if isinstance(start_time, datetime.datetime) & isinstance(end_time, datetime.datetime):
        if end_time < start_time:
            if verbose: print("Exchanged start and end...")
            end_time, start_time = start_time, end_time
        print("Time range = [%s, %s]" % (str(start_time), str(end_time)))
        if verbose: 
            print("Interval Length = %s" % (str(end_time - start_time)))
        else: pass
    else:
        raise TypeError(
            "start_time and end_time must be datetime.datetime, current type: %s and %s" %(str(type(start_time)), str(type(end_time)))
            )

    # set keys
    keys0 = ['ephem', 'mag_rtn', 'mag_1min_rtn', 'mag_burst_rtn', 'mag_srf', 'mag_burst_srf', 'swa']
    if keys is None:
        keys = keys0
    else:
        for key in keys:
            if key in keys0: pass
            else: raise ValueError(
                "key: ''%s'' not supported! Supported keys: \n" %(key) 
                + 
                "''%s'' \n"*len(keys0) %(tuple(keys0))
            )

    # dict for SPDF keys
    spdf_infos = {
        'ephem': {
            'key': 'SOLO_HELIO1DAY_POSITION',
            'vars': ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_rtn': {
            'key': 'SOLO_L2_MAG-RTN-NORMAL',
            'vars': ['B_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_1min_rtn': {
            'key': 'SOLO_L2_MAG-RTN-NORMAL-1-MINUTE',
            'vars': ['B_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_burst_rtn': {
            'key': 'SOLO_L2_MAG-RTN-BURST',
            'vars': ['B_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_srf': {
            'key':'SOLO_L2_MAG-SRF-NORMAL',
            'vars': ['B_SRF'],
            'status': None,
            'data': None,
            'dataframe': None
        },
        'mag_burst_srf': {
            'key': 'SOLO_L2_MAG-SRF-BURST',
            'vars': ['B_SRF'],
            'status': None,
            'data': None,
            'dataframe': None
            },
        'swa': {
            'key': 'SOLO_L2_SWA-PAS-GRND-MOM',
            'vars': ['unrecovered_count','total_count','quality_factor','N','V_SRF','V_RTN','P_SRF','P_RTN','TxTyTz_SRF','TxTyTz_RTN','T','V_SOLO_RTN'],
            'status': None,
            'data': None,
            'dataframe': None
        }
    }

    if get_keys:
        return spdf_infos

    # ------ initialize dictionary ------ #
    spdf_data = {
        'spacecraft'    :   'SOLO',
        'sc'            :   1,
        'start_time'    :   start_time,
        'end_time'      :   end_time,
        'spdf_infos'    :   {}
        }
    
    # ------ download data ------ #
    time = [start_time, end_time]
    for key in keys:
        if verbose: print("Loading %s from CDAWEB..." %(spdf_infos[key]['key']))
        status, data = cdas.get_data(spdf_infos[key]['key'], spdf_infos[key]['vars'], time[0], time[1])
        spdf_info = spdf_infos[key]
        spdf_info['status'] = status
        spdf_info['data'] = data
        spdf_data['spdf_infos'][key] = spdf_info

    if verbose: print("Done. Processing Data...")

    # ------ create dataframe ------ #
    # ephem
    try:
        key = 'ephem'
        if key in keys:
            df = pd.DataFrame(spdf_data['spdf_infos'][key]['data']).set_index("Epoch")
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_rtn
    try:
        key = 'mag_rtn'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['EPOCH'],
                    data = spdf_data['spdf_infos'][key]['data']['B_RTN'],
                    columns = ['Br','Bt','Bn']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_1min_rtn
    try:
        key = 'mag_1min_rtn'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['EPOCH'],
                    data = spdf_data['spdf_infos'][key]['data']['B_RTN'],
                    columns = ['Br','Bt','Bn']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))    

    # mag_burst_rtn
    try:
        key = 'mag_burst_rtn'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['EPOCH'],
                    data = spdf_data['spdf_infos'][key]['data']['B_RTN'],
                    columns = ['Br','Bt','Bn']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_srf
    try:
        key = 'mag_srf'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['EPOCH'],
                    data = spdf_data['spdf_infos'][key]['data']['B_SRF'],
                    columns = ['Bx','By','Bz']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # mag_srf
    try:
        key = 'mag_burst_srf'
        if key in keys:
            df = pd.DataFrame(
                    index = spdf_data['spdf_infos'][key]['data']['EPOCH'],
                    data = spdf_data['spdf_infos'][key]['data']['B_SRF'],
                    columns = ['Bx','By','Bz']
            )
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))

    # swa
    try:
        key = 'swa'
        if key in keys:
            df = pd.DataFrame(index = spdf_data['spdf_infos']['swa']['data']['Epoch'])
            df = df.join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['swa']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['swa']['data']['V_SRF'],
                    columns = ['Vx','Vy','Vz']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['swa']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['swa']['data']['V_RTN'],
                    columns = ['Vr','Vt','Vn']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['swa']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['swa']['data']['N'],
                    columns = ['np']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['swa']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['swa']['data']['T'],
                    columns = ['TEMP']
                )
            )
            df['Vth'] = 13.84112218 * df['TEMP'].apply(np.sqrt) # 1eV = 13.84112218 km/s (kB*T = 1/2*mp*Vth^2)
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))


    # span
    try:
        key = 'span'
        if key in keys:
            df = pd.DataFrame(index = spdf_data['spdf_infos']['span']['data']['Epoch'])
            df = df.join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['DENS'],
                    columns = ['np']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['VEL'],
                    columns = ['Vx','Vy','Vz']
                )
            ).join(
                pd.DataFrame(
                    index = spdf_data['spdf_infos']['span']['data']['Epoch'],
                    data = spdf_data['spdf_infos']['span']['data']['TEMP'],
                    columns = ['TEMP']
                ))
            # pre-clean the data
            # get rid of minus temperature
            df.loc[df['TEMP'] < 0, :] = np.nan
            # get rid of 0.0 in temperature
            df.loc[df['TEMP'].abs() < 1e-2, :] = np.nan
            # calculate Vth
            df['Vth'] = 13.84112218 * np.sqrt(df['TEMP']) # Vth [km/s] = 13.84112218 √(TEMP[eV])
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))



    if verbose: print("Done.")

    
    return spdf_data


def LoadTimeSeriesFromSPEDAS(sc, start_time, end_time, rootdir = None, rolling_rate = '1H', 
    settings = None, credentials = None):
    """ 
    Load Time Series with SPEDAS 
    going to find data in the local directory
    Default resample rate for dfmag is 1s, dfts and dfpar are 5s
    Input:
        sc                          int (0: PSP, 1: SOLO)
        start_time,end_time         pd.Timestamp
    Return:
        dfts                        combined dataframe
        dfmag                       magnetic field dataframe
        dfpar                       particle dataframe
        misc                        miscellaneous 
    """

    # check pyspedas location:
    print(pyspedas.__file__)

    if sc == 0:
        dfts, dfmag, dfpar, misc = LoadTimeSeriesFromSPEDAS_PSP(
            sc, start_time, end_time, 
            rootdir = rootdir, 
            rolling_rate = rolling_rate,
            credentials = credentials,
            settings = settings
            )
    elif sc == 1:
        dfts, dfmag, dfpar, misc = LoadTimeSeriesFromSPEDAS_SOLO(
            sc, start_time, end_time, 
            rootdir = rootdir, 
            rolling_rate = rolling_rate,
            settings = settings
        )
    else:
        raise ValueError("sc = %d not supported!" %(sc))

    return dfts, dfmag, dfpar, misc


def LoadTimeSeriesFromSPEDAS_SOLO(sc, start_time, end_time, rootdir = None, rolling_rate = '1H', settings = None):
    """ Load Time Series from SPEDAS with Solar Orbiter """

    # change to root dir
    if rootdir is None:
        pass
    else:
        os.chdir(rootdir)

    # Solar Orbiter
    if sc == 1:

        # check local directory
        if os.path.exists("./solar_orbiter_data"):
            pass
        else:
            raise ValueError("No local data folder is present!")

        t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
        t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

        swadata = pyspedas.solo.swa(trange=[t0, t1], datatype='pas-grnd-mom')
        data = get_data(swadata[0])

        dfpar = pd.DataFrame(
            # index = time_string.time_datetime(time=data.times, tz=None)
            index = data.times
        )
        temp = get_data('N')
        dfpar = dfpar.join(
            pd.DataFrame(
                # index = time_string.time_datetime(time=np.times, tz=None),
                index = temp.times,
                data = temp.y,
                columns = ['np']
            )
        )
        temp = get_data('T')
        dfpar = dfpar.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['T']
            )
        )
        dfpar['Vth'] = 13.84112218 * np.sqrt(dfpar['T']) # 1eV = 13.84112218 km/s (kB*T = 1/2*mp*Vth^2)

        temp = get_data('V_RTN')
        dfpar = dfpar.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vr','Vt','Vn']
            )
        )

        temp = get_data('V_SRF')
        dfpar = dfpar.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vx','Vy','Vz']
            )
        )

        temp = get_data('V_SOLO_RTN')
        dfpar = dfpar.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['sc_vel_r','sc_vel_t','sc_vel_n']
            )
        )

        dfpar.index = time_string.time_datetime(time=dfpar.index)
        dfpar.index = dfpar.index.tz_localize(None)
        dfpar.index.name = 'datetime'


        time = [(pd.Timestamp(t0)-pd.Timedelta('3d')).to_pydatetime(), (pd.Timestamp(t1)+pd.Timedelta('3d')).to_pydatetime()]
        status, data = cdas.get_data('SOLO_HELIO1DAY_POSITION', ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON'], time[0], time[1])

        dfdis = pd.DataFrame(
            index = data['Epoch'],
            data = data[['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']]
        ).resample('5s').interpolate()
        dfdis.index.name = 'datetime'

        # join dfpar and dfdis
        dfpar = dfpar.resample('5s').mean().join(dfdis)
        dfpar['Dist_au'] = dfpar['RAD_AU']

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

        dfts = dfmag.resample('5s').mean().join(
            dfpar.resample('5s').mean())

        dfts['Dist_au'] = dfts['RAD_AU'].interpolate()

        dfts[['Vr0','Vt0','Vn0']] = dfts[['Vr','Vt','Vn']].rolling(rolling_rate).mean()
        dfts[['Vx0','Vy0','Vz0']] = dfts[['Vx','Vy','Vz']].rolling(rolling_rate).mean()
        dfts[['Br0','Bt0','Bn0']] = dfts[['Br','Bt','Bn']].rolling(rolling_rate).mean()
        dfts[['Bx0','By0','Bz0']] = dfts[['Bx','By','Bz']].rolling(rolling_rate).mean()

        dfpar[['Vr0','Vt0','Vn0']] = dfpar[['Vr','Vt','Vn']].rolling(rolling_rate).mean()
        dfpar[['Vx0','Vy0','Vz0']] = dfpar[['Vx','Vy','Vz']].rolling(rolling_rate).mean()
        dfmag[['Br0','Bt0','Bn0']] = dfmag[['Br','Bt','Bn']].rolling(rolling_rate).mean()
        dfmag[['Bx0','By0','Bz0']] = dfmag[['Bx','By','Bz']].rolling(rolling_rate).mean()

        # nothing to be stored for solar orbiter
        misc = {'dfdis': dfdis}

        return dfts, dfmag, dfpar, misc


    else:
        raise ValueError("sc=%d not supported!" %(sc))


def LoadTimeSeriesFromSPEDAS_PSP(sc, start_time, end_time, 
    rootdir = None, rolling_rate = '1H', settings = None, credentials = None):
    """" 
    Load Time Serie From SPEDAS, PSP 
    settings if not None, should be a dictionary, necessary settings:

    spc_only: boolean
    span_only: boolean
    mix_spc_span: dict:
        {'priority': 'spc' or 'span'}
    keep_spc_and_span: boolean

    keep_keys: ['np','Vth','Vx','Vy','Vz','Vr','Vt','Vn']
    
    Note that the priority is using SPAN
    """


    # change to root dir
    if rootdir is None:
        pass
    else:
        os.chdir(rootdir)

    # default settings
    if settings is None:
        settings = {
            'particle_mode': 'empirical',
            'final_freq': '5s',
            'use_hampel': False,
            'interpolate_qtn': True
        }

    # Parker Solar Probe
    if sc == 0:
        # check local directory
        if os.path.exists("./psp_data"):
            pass
        else:
            raise ValueError("No local data folder ./psp_data is present!")

        t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
        t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

        # Quasi-Thermal-Noise
        try:
            # Quasi-Thermal Noise for electron density
            try:    
                qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_v1v2', level='l3', 
                            varnames = [
                                'electron_density',
                                'electron_core_temperature'
                            ], 
                            time_clip=True)
                temp = get_data(qtndata[0])
            except:
                print("No QTN data is presented in the public repository!")
                print("Trying unpublished data... please provide credentials...")
                if credentials is None:
                    raise ValueError("No credentials are provided!")

                username = credentials['psp']['fields']['username']
                password = credentials['psp']['fields']['password']

                qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_V1V2', level='l3', 
                varnames = [
                    'electron_density',
                    'electron_core_temperature'
                ], 
                time_clip=True, username=username, password=password)
                temp = get_data(qtndata[0])


            dfqtn = pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['ne_qtn']
            )

            dfqtn['np_qtn'] = dfqtn['ne_qtn']/1.08 # 4% of alpha particle
            dfqtn.index = time_string.time_datetime(time=dfqtn.index)
            dfqtn.index = dfqtn.index.tz_localize(None)
            dfqtn.index.name = 'datetime'
        except:
            print("No QTN Data!")
            dfqtn = None

        # Magnetic field
        try:
            try:
                names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_rtn_4_per_cycle', level='l2', time_clip=True)
                data = get_data(names[0])
                dfmag1 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag1.columns = ['Br','Bt','Bn']

                names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_sc_4_per_cycle', level='l2', time_clip=True)
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
                    datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                    username=username, password=password
                )
                data = get_data(names[0])
                dfmag1 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag1.columns = ['Br','Bt','Bn']

                names = pyspedas.psp.fields(trange=[t0,t1], 
                    datatype='mag_SC_4_Sa_per_Cyc', level='l2', time_clip=True,
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
        except:
            print("No MAG Data!")
            dfmag = None

        # SPC
        try:
            try:
                spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='l3', 
                                        varnames = [
                                            'np_moment',
                                            'wp_moment',
                                            'vp_moment_RTN',
                                            'vp_moment_SC',
                                            'sc_pos_HCI',
                                            'sc_vel_HCI',
                                            'carr_latitude',
                                            'carr_longitude'
                                        ], 
                                        time_clip=True)

                data = get_data(spcdata[0])
            except:
                print("No SPC data is presented in the public repository!")
                print("Trying unpublished data... please provide credentials...")
                if credentials is None:
                    raise ValueError("No credentials are provided!")

                username = credentials['psp']['sweap']['username']
                password = credentials['psp']['sweap']['password']

                spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='L3', 
                                        varnames = [
                                            'np_moment',
                                            'wp_moment',
                                            'vp_moment_RTN',
                                            'vp_moment_SC',
                                            'sc_pos_HCI',
                                            'sc_vel_HCI',
                                            'carr_latitude',
                                            'carr_longitude'
                                        ], 
                                        time_clip=True, username=username, password=password)

                data = get_data(spcdata[0])

            dfspc = pd.DataFrame(
                # index = time_string.time_datetime(time=data.times, tz=None)
                index = data.times
            )

            temp = get_data(spcdata[0])
            dfspc = dfspc.join(
                pd.DataFrame(
                    # index = time_string.time_datetime(time=np.times, tz=None),
                    index = temp.times,
                    data = temp.y,
                    columns = ['np']
                )
            )

            temp = get_data(spcdata[1])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Vth']
                )
            )

            temp = get_data(spcdata[2])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Vr','Vt','Vn']
                )
            )

            temp = get_data(spcdata[3])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Vx','Vy','Vz']
                )
            )

            temp = get_data(spcdata[4])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['sc_x','sc_y','sc_z']
                )
            )

            temp = get_data(spcdata[5])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['sc_vel_x','sc_vel_y','sc_vel_z']
                )
            )

            temp = get_data(spcdata[6])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['carr_lat']
                )
            )

            temp = get_data(spcdata[7])
            dfspc = dfspc.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['carr_lon']
                )
            )

            # calculate Dist_au
            dfspc['Dist_au'] = (dfspc[['sc_x','sc_y','sc_z']]**2).sum(axis=1).apply(np.sqrt)/au_to_km

            dfspc.index = time_string.time_datetime(time=dfspc.index)
            dfspc.index = dfspc.index.tz_localize(None)
            dfspc.index.name = 'datetime'
        except:
            print("No SPC Data!")
            dfspc = None

        # SPAN
        try:
            try:
                spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00_l3_mom', level='l3', 
                        varnames = [
                            'DENS',
                            'VEL_SC',
                            'VEL_RTN_SUN',
                            'TEMP',
                            'SUN_DIST',
                            'SC_VEL_RTN_SUN'
                        ], 
                        time_clip=True)
                temp = get_data(spandata[0])
            except:
                print("No SPAN data is presented in the public repository!")
                print("Trying unpublished data... please provide credentials...")
                if credentials is None:
                    raise ValueError("No credentials are provided!")

                username = credentials['psp']['sweap']['username']
                password = credentials['psp']['sweap']['password']

                spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00', level='L3', 
                        varnames = [
                            'DENS',
                            'VEL_SC',
                            'VEL_RTN_SUN',
                            'TEMP',
                            'SUN_DIST',
                            'SC_VEL_RTN_SUN'
                        ], 
                        time_clip=True, username=username, password=password)
                temp = get_data(spandata[0])


            dfspan = pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['np']
            )

            temp = get_data(spandata[1])
            dfspan = dfspan.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Vx', 'Vy', 'Vz']
                )
            )

            temp = get_data(spandata[2])
            dfspan = dfspan.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Vr', 'Vt', 'Vn']
                )
            )

            temp = get_data(spandata[3])
            dfspan = dfspan.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['TEMP']
                )
            )

            temp = get_data(spandata[4])
            dfspan = dfspan.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['Dist_au']
                )
            )
            dfspan['Dist_au'] = dfspan['Dist_au']/au_to_km

            temp = get_data(spandata[5])
            dfspan = dfspan.join(
                pd.DataFrame(
                    index = temp.times,
                    data = temp.y,
                    columns = ['sc_vel_r','sc_vel_t','sc_vel_n']
                )
            )

            # calculate Vth from TEMP
            # k T (eV) = 1/2 mp Vth^2 => Vth = 13.84112218*sqrt(TEMP)
            dfspan['Vth'] = 13.84112218 * np.sqrt(dfspan['TEMP'])

            # for span the thermal speed is defined as the trace, hence have a sqrt(3) different from spc
            dfspan['Vth'] = dfspan['Vth']/np.sqrt(3)

            dfspan.index = time_string.time_datetime(time=dfspan.index)
            dfspan.index = dfspan.index.tz_localize(None)
            dfspan.index.name = 'datetime'
        except:
            print("No SPAN!")
            dfspan = None

        # merge particle data

        if 'particle_mode' in settings.keys():
            parmode = settings['particle_mode']
        else:
            parmode = 'empirical'

        # create empty dfpar with index
        freq = settings['final_freq']
        index = pd.date_range(
            start = start_time, 
            end = end_time, 
            freq = freq
        )
        dfpar = pd.DataFrame(
            index = index
        )

        print("Parmode: %s" %(parmode))
        # fill dfpar with values
        if parmode == 'spc_only':
            dfpar = dfpar.join(dfspc.resample(freq).mean())
        elif parmode == 'span_only':
            dfpar = dfpar.join(dfspan.resample(freq).mean())
        elif parmode == 'empirical':
            # empirical use of data
            # encounter date: https://sppgway.jhuapl.edu/index.php/encounters
            # before encounter 9 (Perihelion: 2021-08-09/19:11) use SPC for solar wind speed
            # at and after encounter 8, mix SPC and SPAN for solar wind speed
            # prioritize QTN for density, and fill with SPC, and with SPAN
            
            # proton density with QTN
            dfpar = dfpar.join(dfqtn.resample(freq).mean())

            # the density is not very fluctuating, hence interpolate
            try:
                if settings['interpolate_qtn']:
                    print("QTN is interpolated!")
                    dfpar['ne_qtn'] = dfpar['ne_qtn'].interpolate()
            except:
                pass

            dfpar['np'] = dfpar['ne_qtn']/1.08

            keep_keys = ['Vx','Vy','Vz','Vr','Vt','Vn','Vth','Dist_au']

            # if no SPC or no SPAN data:
            if dfspc is None:
                dfspc = pd.DataFrame(index = dfpar.index)
                dfspc[keep_keys] = np.nan

            if dfspan is None:
                dfspan = pd.DataFrame(index = dfpar.index)
                dfspan[keep_keys] = np.nan


            # Perihelion cut
            ind1 = dfpar.index < pd.Timestamp('2021-07-15')
            ind2 = dfpar.index >= pd.Timestamp('2021-07-15')

            ind11 = dfspc.index < pd.Timestamp('2021-07-15')
            ind12 = dfspc.index >= pd.Timestamp('2021-07-15')

            ind21 = dfspan.index < pd.Timestamp('2021-07-15')
            ind22 = dfspan.index >= pd.Timestamp('2021-07-15')

            # before encounter 9 use spc
            dfpar1 = dfspc.loc[ind11,keep_keys].resample(freq).mean()

            # use span after 2021-07-15
            dfpar2 = dfspan.loc[ind22,keep_keys].resample(freq).mean()
            # dfpar2 = dfpar21.copy()
            # ind21t = dfpar21['Vr'].apply(np.isnan)
            # ind22t = dfpar22['Vr'].apply(np.isnan)
            # dfpar2.loc[~ind21t,['MOMENT_FLAG','DENSITY_FLAG']] = 1 # SPC
            # dfpar2.loc[~ind22t,['MOMENT_FLAG','DENSITY_FLAG']] = 2 # SPAN
            # fill SPAN empty with SPC
            # dfpar2.loc[ind21t] = dfpar22.loc[ind21t]

            # # combine qtn
            # dfpar_qtn = pd.DataFrame(
            #     index = index
            # ).join(dfqtn.resample(freq).mean())
            
            # merge qtn and particle data
            dftemp = pd.concat([dfpar1,dfpar2])
            dfpar = dfpar.join(dftemp[keep_keys])

        elif parmode == 'keep_all':
            raise ValueError("particle mode: %s under construction!" %(parmode))
        
        else:
            raise ValueError("particle mode: %s not supported!" %(parmode))


        if settings['use_hampel'] == True:
            for k in dfpar.columns:
                ns, _ = hampel_filter_forloop_numba(dfpar[k].values, 100)
                dfpar[k] = ns




        # create dfts
        dfts = dfmag.resample(freq).mean().join(
            dfpar.resample(freq).mean()
        )

        dfts[['Vr0','Vt0','Vn0']] = dfts[['Vr','Vt','Vn']].rolling(rolling_rate).mean().interpolate()
        dfts[['Vx0','Vy0','Vz0']] = dfts[['Vx','Vy','Vz']].rolling(rolling_rate).mean().interpolate()
        dfts[['Br0','Bt0','Bn0']] = dfts[['Br','Bt','Bn']].rolling(rolling_rate).mean().interpolate()
        dfts[['Bx0','By0','Bz0']] = dfts[['Bx','By','Bz']].rolling(rolling_rate).mean().interpolate()

        # resample dfmag to create B?0
        dfmag = dfmag.resample(freq).mean()
        dfmag[['Br0','Bt0','Bn0','Bx0','By0','Bz0']] = dfmag[['Br','Bt','Bn','Bx','By','Bz']].rolling(rolling_rate).mean()
        dfpar = dfts[['Vr','Vt','Vn','Vx','Vy','Vz','Vr0','Vt0','Vn0','Vx0','Vy0','Vz0','np','Vth']]

        misc = {
            'dfqtn': dfqtn,
            'dfspc': dfspc,
            'dfspan': dfspan,
            'parmode': parmode,
            'settings': settings
        }

        return dfts, dfmag, dfpar, misc
    else:
        raise ValueError("sc = %d, wrong function!!" %(sc))


def LoadSCAMFromSPEDAS_PSP(start_time, end_time, credentials = None):
    """ 
    load scam data with pyspedas and return a dataframe
    Input:
        start_time, end_time                pd.Timestamp
        (optional) credentials              dictionary, {'username':..., 'password':...}
    Output:
        return None if no data is present, otherwise a dataframe containing all the scam data
    """

    # check pyspedas
    if os.path.exists(Path(".").absolute().parent.joinpath("pyspedas")):
        print("Using pyspedas at %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))
    else:
        raise ValueError("Please clone pyspedas to %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))

    t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

    if credentials is None:
        scam_vars = pyspedas.psp.fields(
            trange=[t0, t1], datatype='merged_scam_wf', level='l3', time_clip=True, downloadonly = False
        )

        if scam_vars == []:
            return None

        data = get_data(scam_vars[0])
        dfscam = pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Bu','Bv','Bw']
        )

        data = get_data(scam_vars[1])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Bx','By','Bz']
            )
        )

        data = get_data(scam_vars[2])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Br','Bt','Bn']
            )
        )

        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'

        return dfscam

    else:
        # have credentials

        try:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf', level='l3', time_clip=True, downloadonly = False,
                username = credentials['username'], password = credentials['password']
            )
        except:
            raise ValueError('Wrong Username or Password!')

        if scam_vars == []:
            return None

        data = get_data(scam_vars[0])
        dfscam = pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Bu','Bv','Bw']
        )

        data = get_data(scam_vars[1])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Bx','By','Bz']
            )
        )

        data = get_data(scam_vars[2])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Br','Bt','Bn']
            )
        )

        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'

        return dfscam



# Loading Procedures



# -----------  Tools ----------- #

def FindIntervalInfo(sc, start_time, end_time, verbose = False, spdf = False, local = False, tsv = None, keys = None):

    if spdf:
        spdf_data = LoadTimeSeriesSPDF(sc = sc, 
            start_time=start_time.to_pydatetime(), 
            end_time = end_time.to_pydatetime(), 
            verbose = verbose, keys = keys)

        d = {
            'sc': sc,
            'start_time': start_time,
            'end_time': end_time,
            'spdf_data': spdf_data,
            'Diagnostics': None
        }
        return d

    if local:
        try:
            if sc == tsv.sc:
                pass
            else:
                raise ValueError("sc = %d must equal tsv.sc = %d" %(sc, tsv.sc))
        except:
            raise ValueError("tsv must be a TimeSeriesViewer Object!!!")

        dfts = tsv.ExportTimeSeries(start_time, end_time, verbose = verbose)

        d = {
            'sc': sc,
            'start_time': start_time,
            'end_time': end_time,
            'local_data': {
                'dfts_raw': tsv.dfts_raw
            },
            'dfts': dfts,
            'Diagnostics': None
        }
        return d




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

    freqs = np.fft.rfftfreq(len(x), dt)
    # freqs = freqs[freqs>0]
    # idx   = np.argsort(freqs)
    
    return freqs, B_pow


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
def smoothing_function_obsolete(x,y, window=2, pad = 1):
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



def curve_fit_log_wrap(x, y, x0, xf):  
    # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]

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

def UpdatePSDDict(path, credentials = None):
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
        credentials = credentials
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

    else:
        raise ValueError("sc=%d not supported!" %(sc))

    # determine resample frequency
    val, counts = np.unique(np.diff(dfmag.index), return_counts=True)

    # gaps are not counted as real cadence hence remove gap jumps
    # val are in timedelta64[ns], cadance > 5s are regarded as gaps
    val = val[val.astype(float) <= 5e9]

    # find the maximum of the cadence, and resample to the corresponding freq
    if (np.max(val).astype(float)/1e6) < 100:
        # 10 ms
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