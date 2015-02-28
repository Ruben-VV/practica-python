import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *
from math import factorial
##from scipy.stats import norm, t, chi, sem
from scipy.stats import *

import pandas as pd


##import scipy.stats



pd.Series(range(1,10,2))

##media - z_a/2*s/sqrt(n),  media + z_a/2*s/sqrt(n)

##z_a/2 = norm.ppf(a/2)
##a/2 = norm.cdf(z_a/2)


def icnorm_media(x, a=0.05, sigma=None):
    """Calculo de Intervalos de Confianza
    IC para distribucion normal, seleccionando casos dependiendo si conocemos
    la desviacion poblacional sigma (o la varianza var = sigma**2
    """
##    import numpy as np, pandas as pd
##    from scipy.stats import norm, t, chi, sem
##    import scipy.stats
    n = len(x)
    if type(x) == tuple or type(x) == list:
        x = array(x)
    if sigma == None:
        s = sigma
        z = -norm.ppf(a/2.)
    else:
        s = x.std(ddof=1)
        if n > 30:
            z = -norm.ppf(a/2.)
        else:
            z = -t.ppf(a/2.,n-1)
            z = t.ppf((1+1-a)/2,n-1)
##scipy.stats.sem(x) calcula el error estandar que es igual a std/sqrt(n)
##    m, se = np.mean(x), scipy.stats.sem(x)
##    h = se * z
    s = std(x,ddof=1)
    if isinstance(x,pd.DataFrame):
        m = x.mean().values
##        media = x.mean().values.tolist()
    else:
         m = mean(x)  
    h = s / sqrt(n) * z
##    ic = pd.DataFrame([x-h,x+h], index=['ICm1', 'ICm2'], columns=['P','log P'])

##    return ic
    return m, h, m-h, m+h
##        media = x.mean().values.tolist()


