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


def LoadPSPTimeSeriesSPDF(start_time, end_time, keys = None, verbose = True):
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
    span                PSP_SWP_SPI_SF00_L3_MOM_INST    SPAN Moments                                ['Vx','Vy','Vz','np','Vth']

    Input:
        start_time  [datetime.datetime]   :   start of the interval
        end_time    [datetime.datetime]   :   end of the interval

    Keywords:
        keys = None                             can be a list containing all the desired keys

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
    keys0 = ['ephem', 'mag_rtn', 'mag_sc', 'mag_1min_rtn', 'mag_1min_sc', 'mag_scm', 'spc', 'span']
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
            indnan = spdf_data['spdf_infos']['span']['data']['QUALITY_FLAG'] == int('0000000000000000', 2)
            df.loc[indnan,:] = np.nan
            # get rid of minus temperature
            df.loc[df['TEMP'] < 0, :] = np.nan
            # get rid of 0.0 in temperature
            df.loc[df['TEMP'].abs() < 1e-2, :] = np.nan
            # calculate Vth
            df['Vth'] = 13.84112218 * df['TEMP'] # 1eV = 13.84112218 km/s (kB*T = 1/2*mp*Vth^2)
            df.index.name = 'datetime'
            spdf_data['spdf_infos'][key]['dataframe'] = df
    except:
        print("Processing %s (%s) failed!" %(key, spdf_infos[key]['key']))



    if verbose: print("Done.")

    
    return spdf_data


def LoadSolOTimeSeriesSPDF(start_time, end_time, keys = None, verbose = True):
    """
    Load Solar Orbiter Time Series, this is a wrapper for SPDF API
    All Time Series will be stored with dataframe, NOT resampled
    Keys available for loading:
    Key:                SPDF Key:                       Description:                                Columns:     
    ephem               SOLO_HELIO1DAY_POSITION         Ephemeris                                   ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']                                               
    mag_rtn             PSP_FLD_L2_MAG_RTN              Full resolution magnetic field              ['Br','Bt','Bn']                                        
    mag_sc              PSP_FLD_L2_MAG_SC               Full resolution sc frame magnetic field     ['Bx','By','Bz']
    mag_1min_rtn        PSP_FLD_L2_MAG_RTN_1MIN         1min RTN magnetic field                     ['Br','Bt','Bn']
    mag_1min_sc         PSP_FLD_L2_MAG_SC_1MIN          1min SC magnetic field                      ['Bx','By','Bz']
    mag_scm             PSP_FLD_L3_MERGED_SCAM_WF       Merged FIELDS and SCAM data                 *Keep as dict
    swa                 SOLO_L2_SWA-PAS-GRND-MOM        SWA Moments                                 ['Vx','Vy','Vz','np','Vth','Vr','Vt','Vn']   
    span                PSP_SWP_SPI_SF00_L3_MOM_INST    SPAN Moments                                ['Vx','Vy','Vz','np','Vth']

    Input:
        start_time  [datetime.datetime]   :   start of the interval
        end_time    [datetime.datetime]   :   end of the interval

    Keywords:
        keys = None                             can be a list containing all the desired keys

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
    keys0 = ['ephem', 'mag_rtn', 'mag_sc', 'mag_1min_rtn', 'mag_1min_sc', 'mag_scm', 'spc', 'span']
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
    if R < 0.3:
        T = (R/0.3) * 24
    elif ((R >= 0.3) & (R < 1.0)):
        T = (R/0.3)**0.3 * 24
    else:
        T = (1.0/0.3)**0.3 * (R/1.0)**0.27 * 24

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

    freqs = np.fft.fftfreq(len(x), dt)
    freqs = freqs[freqs>0]
    idx   = np.argsort(freqs)
    
    return freqs[idx], B_pow[idx]


    