import pandas as pd
import numpy as np
import pyspedas
from pytplot import get_data
from pyspedas.utilities import time_string
import pandas as pd
import numpy as np

from numba import jit,njit,prange
import numba

@jit(nopython=True, parallel=True)
def StructureFunctionDistribution(arr, s, n):
    """
    Calculate the n th order structure function distribution of arr on scale s
    Input:
        arr: np.array
        s: int, for scale
        n: int, order
     Output:
        SFD = (arr(t+s) - arr(t))^n
    """
    SFD = np.zeros([l-s-1])
    l = len(arr)

    for i1 in range(l-s-1):
        SFD[i1] = (arr(i1+s) - arr(i1))**n

    return SFD
