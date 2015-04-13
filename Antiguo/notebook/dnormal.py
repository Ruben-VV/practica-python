#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
from matplotlib import pylab, mlab, pyplot
import matplotlib.pyplot as plt
#plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

#from pylab import *
from numpy import *
import numpy.random as npr
from math import factorial
import scipy.misc
##from scipy.stats import norm, t, chi2, sem
from scipy.stats import *

import pandas as pd


def icnorm_media(x, a=0.05, sigma=None):
    """Calculo de Intervalos de Confianza
    IC para distribucion normal, seleccionando casos dependiendo si conocemos
    la desviacion poblacional sigma (o la varianza var = sigma**2
    a -> nivel de significacion
    c = 1-a -> nivel de confianza
    """
    n = len(x)
    if type(x) == tuple or type(x) == list:
        x = 1.0*array(x)
    if sigma != None:
        s = sigma
        z = -norm.ppf(a/2.)
    else:
        s = x.std(ddof=1)
        if n > 30:
            z = -norm.ppf(a/2.)
        else:
            z = -t.ppf(a/2.,n-1)
##            z2 = t.ppf((1+1-a)/2,n-1)
##scipy.stats.sem(x) calcula el error estandar que es igual a std/sqrt(n)
##    m, se = np.mean(x), scipy.stats.sem(x)
##    h = se * z
##    se = sem(x)
##    h2 = se * z2
    s = std(x,ddof=1)
    if isinstance(x,pd.DataFrame):
        m = x.mean().values
##        media = x.mean().values.tolist()
    else:
         m = mean(x)  
    h = s * z / sqrt(n)
    if isinstance(x, pd.DataFrame):
        ic = pd.DataFrame([m, m-h,m+h], index=['media', 'ICm1', 'ICm2'],
                      columns=x.columns)
##        ic = pd.DataFrame([m-h,m+h], index=['ICm1', 'ICm2'],
##                          columns=x.columns)
##        m = pd.DataFrame([m], index=['media'], columns=x.columns)
    else:
        ic = pd.DataFrame([m, m-h,m+h], index=['media', 'ICm1', 'ICm2'],
                          columns=[str((1-a)*100) + '%'])
##        ic = pd.DataFrame([m-h,m+h], index=['ICm1', 'ICm2'],
##                          columns=[str((1-a)*100) + '%'])
##    return m, h, m-h, m+h #, h2, m-h2, m+h2
    return ic
##    return m, ic
##        media = x.mean().values.tolist()

def icnorm_var(x, a=0.05):
    """Calculo de Intervalos de Confianza
    IC para la varianza y la desviacion estandar usando la distribución de Pearson o chi2
    a -> nivel de significacion
    c = 1-a -> nivel de confianza
    """
    n = len(x)
    if type(x) == tuple or type(x) == list:
        x = 1.0*array(x)      
    if isinstance(x,pd.DataFrame):
        s = x.std(ddof=1).values
        var = x.var(ddof=1).values
    else:
        s = std(x,ddof=1)
        var = x.var(ddof=1)        
    z1, z2 = chi2.ppf(1-a/2,n-1), chi2.ppf(a/2,n-1)   
    i1, i2 = (n-1) * s**2 / z1, (n-1) * s**2 / z2    
    if isinstance(x, pd.DataFrame):
        ic = pd.DataFrame([var, i1, i2, s, sqrt(i1), sqrt(i2)],
                          index=['Varianza', 'ICv1', 'ICv2', 'Desviacion',
                                 'ICstd1', 'ICstd2'], columns=x.columns)
    else:
        ic = pd.DataFrame([var, i1, i2, s, sqrt(i1), sqrt(i2)],
                          index=['Varianza', 'ICv1', 'ICv2', 'Desviacion',
                                 'ICstd1', 'ICstd2'],
                          columns=[str((1-a)*100) + '%'])
    return ic
