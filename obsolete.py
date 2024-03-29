# -----------  obsolete ----------- #

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

