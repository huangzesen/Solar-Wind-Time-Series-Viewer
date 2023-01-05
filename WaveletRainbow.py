import pandas as pd
import numpy as np

from StructureFunctions import MagStrucFunc
from TSUtilities import curve_fit_log_wrap, trace_PSD_wavelet

def WaveletRainbow(
    dfmag, res, N = 10, win_ratio = 0.8, verbose = True
):
    """
    This function calculate N wavelet spectra from given dfmag
    by sliding through the dfmag with a window sized win_ratio.
    
    Input:
    ---------------
    dfmag : pandas dataframe
        consists of Bx, By, Bz and timed index
    res : float
        resolution of dfmag, in seconds
    
    keywords:
    ---------------
    N : int
        number of sub-intervals
    win_ratio : float, valued [0,1]
        normalized length of the sub-intervals
    """

    # drop na
    dfmag = dfmag.dropna()

    # the diagnostics will be stored in diags
    diags = {
        'N': N,
        'win_ratio': win_ratio,
        'resolution': res
    }

    # calculate the start point
    start = int(np.floor(len(dfmag)*win_ratio))
    step = int(np.floor((len(dfmag) - start)/N))

    w1s = [step*n for n in range(N)]
    w2s = [start+step*n for n in range(N)]


    for i1 in range(N):
        Br = dfmag['Bx'][w1s[i1]:w2s[i1]]
        Bt = dfmag['By'][w1s[i1]:w2s[i1]]
        Bn = dfmag['Bz'][w1s[i1]:w2s[i1]]

        _,_,_,freqs_wl,PSD_wl,scales,coi = trace_PSD_wavelet(
            Br.values, Bt.values, Bn.values, res, 
            dj = 1./12
        )

        if verbose:
            print("WL rainbow: %d out of %d finished" %(i1+1, N))
        
        
        t0 = Br.index[0]
        t1 = Br.index[-1]
        maxtime = np.log10((t1-t0)/pd.Timedelta('1s')/2)
        mintime = np.log10(2*res)
        struc_funcs = MagStrucFunc(Br, Bt, Bn, (mintime, maxtime), 200)
        
        diags['%d' %(i1)] = {
            'freqs': freqs_wl,
            'PSD': PSD_wl,
            'scales': scales,
            'coi': coi,
            'struc_funcs': struc_funcs,
            'struc_funcs_settings': {
                'mintime': mintime,
                'maxtime': maxtime,
                'npts': 200
            }
        }

    return diags
