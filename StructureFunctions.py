import pandas as pd
import numpy as np
import pyspedas
from pytplot import get_data
from pyspedas.utilities import time_string
import pandas as pd
import numpy as np

# from TimeSeriesViewer import TimeSeriesViewer


def CalculateMagDiagnosticsWrapper(sc, tstart, tend, freq_range = (-4,10), credentials = None, ndts = 100):

    #load sigma_c

    tsv = TimeSeriesViewer(sc = sc, start_time_0 = tstart, end_time_0 = tend, credentials = credentials)
    tsv.InitFigure(tstart, tend, no_plot=True, resample_rate = '30s')

    #load mag
    t0 = tstart.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = tend.strftime("%Y-%m-%d/%H:%M:%S")
    try:
        names = pyspedas.psp.fields(trange=[t0,t1], 
            datatype='mag_rtn', level='l2', time_clip=True
        )
    except:
        names = pyspedas.psp.fields(trange=[t0,t1], 
            datatype='mag_RTN', level='l2', time_clip=True,
            username = credentials['psp']['fields']['username'], password = credentials['psp']['fields']['password']
        )
    data = get_data(names[0])
    dfmag = pd.DataFrame(
        index = data[0],
        data = data[1]
    )
    dfmag.columns = ['Br','Bt','Bn']
    dfmag.index = time_string.time_datetime(time=dfmag.index)
    dfmag.index = dfmag.index.tz_localize(None)
    dfmag = dfmag.resample('100ms').mean()

    # dts in ms
    dts = np.floor(1000*np.logspace(freq_range[0], freq_range[1], ndts, base=2))
    dBvecs = []
    dBvecnorms = []
    dBmods = []
    dBmodnorms = []

    for i1 in range(len(dts)):
        dt = pd.Timedelta("%dms" %(int(dts[i1])))

        rolling_window = "%dms" %(int(dts[i1]))

        dBvec, dBvecnorm, dBmod, dBmodnorm = CalculateMagDiagnostics(
            dfmag['Br'], dfmag['Bt'], dfmag['Bn'],
            dt, rolling_window
        )
        dBvecs.append(dBvec)
        dBvecnorms.append(dBvecnorm)
        dBmods.append(dBmod)
        dBmodnorms.append(dBmodnorm)
        
    d = {
        'sc': 0,
        'start_time': tstart,
        'end_time': tend,
        'dfts': tsv.dfts,
        'dfdis': tsv.dfdis,
        'MagDiagnostics': {
            'rolling_window': '10min',
            'dts': dts,
            'dBvecs': dBvecs,
            'dBvecnorms': dBvecnorms,
            'dBmods': dBmods,
            'dBmodnorms': dBmodnorms
        }
    }

    return d

def MagStrucFunc(Br, Bt, Bn, scale_range, npts):
    """
    return the magnetic structure functions
    scale_range in log10 space in seconds
    """

    dBvecs = np.array([])
    dBvecnorms = np.array([])
    dBmods = np.array([])
    dBmodnorms = np.array([])

    dts = np.floor(1000*np.logspace(scale_range[0], scale_range[1], npts, base = 10))
    for i1 in range(len(dts)):
        dt = pd.Timedelta("%dms" %(dts[i1]))
        rolling_window = "%dms" %(dts[i1])
        dBvec, dBvecnorm, dBmod, dBmodnorm = CalculateMagDiagnostics(
            Br, Bt, Bn, dt, rolling_window
        )

        dBvecs = np.append(dBvecs,dBvec)
        dBvecnorms = np.append(dBvecnorms, dBvecnorm)
        dBmods = np.append(dBmods, dBmod)
        dBmodnorms = np.append(dBmodnorms, dBmodnorm)

    results = {
        'dts': dts,
        'dBvecs': dBvecs,
        'dBvecnorms': dBvecnorms,
        'dBmods': dBmods,
        'dBmodnorms': dBmodnorms
    }

    return results

    


def CalculateMagDiagnostics(Br, Bt, Bn, dt, rolling_window):
    """
    Calculate <dBvec> and <dBvec/|B|> of given dt
    """

    dBr, Brmean = TimeseriesDifference(Br, dt, rolling_window)
    dBt, Btmean = TimeseriesDifference(Bt, dt, rolling_window)
    dBn, Bnmean = TimeseriesDifference(Bn, dt, rolling_window)
    dBmod, Bmodmean = TimeseriesDifference(np.sqrt(Br**2+Bt**2+Bn**2), dt, rolling_window)

    dBvecmod = (np.sqrt(dBr**2+dBt**2+dBn**2)).to_numpy()
    Bvecmod = (np.sqrt(Brmean**2+Btmean**2+Bnmean**2)).to_numpy()

    return np.nanmean(dBvecmod), np.nanmean(dBvecmod/Bmodmean), np.nanmean(np.abs(dBmod)), np.nanmean(np.abs(dBmod)/dBvecmod)

def TimeseriesDifference(ts, dt, rolling_window):
    """
    Input:
        ts: pandas timeseries with index of time, increment should be uniform
        dt: pandas time delta
    Output:
        tsout: pandas timeseries with index of time
            tsout(t, dt) = ts(t+dt) - ts(t)
    """

    if not isinstance(ts, pd.Series):
        raise ValueError("ts must be a pandas series")

    if not isinstance(dt, pd.Timedelta):
        raise ValueError("dt must be pandas.Timedelta")

    if not isinstance(ts.index[0], pd.Timestamp):
        raise ValueError("ts.index must be pandas.Timestamp")

    ind1 = ts.index < (ts.index[-1]-dt)

    ind2 = ts.index > (ts.index[0]+dt)

    ts1 = ts[ind1]
    ts2 = ts[ind2]

    if len(ts1) > len(ts2):
        ts1 = ts1[len(ts1)-len(ts2):]
        tsmean = (ts.rolling(rolling_window, center=True).mean().interpolate())[ind2]
    elif len(ts1) < len(ts2):
        ts2 = ts2[0:len(ts1)]
        tsmean = (ts.rolling(rolling_window, center=True).mean().interpolate())[ind1]
    else:
        tsmean = (ts.rolling(rolling_window, center=True).mean().interpolate())[ind1]


    tsdiff = pd.Series(
        index = ts1.index,
        data = ts2.values - ts1.values
    )


    return tsdiff, tsmean

def MovingFitOnSpectrum(x, y, window, limits, pad):
    """
    Return the fit index of the moving window
    """
    yindex = []
    xloc = []
    x1 = limits[0]
    while x1+window < limits[1]:
        ind = (x>=x1) & (x<=x1+window)
        xfit = x[ind]
        yfit = y[ind]

        yindex.append(np.polyfit(xfit, yfit, 1)[0])
        xloc.append(x1+window/2)
        x1 += pad

    return xloc, yindex
    