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
##from scipy.stats import norm
from scipy.stats import *
from rangosuma import *
from bootstrap_test import *
from IC import *

from pandas import *
import pandas as pd


#Jannuary precipitation at Ithaca, NY, 1933-1982, inches
P = [0.44, 1.18, 2.69, 2.08, 3.66, 1.72, 2.82, 0.72, 1.46, 1.30, 1.35, .54,
     2.74, 1.13, 2.5, 1.72, 2.27, 2.82, 1.98, 2.44, 2.53, 2.0, 1.12, 2.13,
     1.36, 4.9, 2.94, 1.75, 1.69, 1.88, 1.31, 1.76, 2.17, 2.38, 1.16, 1.39,
     1.36, 1.03, 1.11, 1.35, 1.44, 1.84, 1.69,
     3.0, 1.36, 6.37, 4.55, .52, .87, 1.51]
year = range(1933,1982+1,1)

ithaca = pd.DataFrame(P,columns=['P'],index=year)
##ithaca.dtypes

ithaca['log P'] = log(ithaca['P'])
slnx = std(ithaca['log P'],1)
##ithaca['log P'].std(ddof=1)
##ithaca['log P'].std(0,1)  #por columnas
##ithaca['log P'].std(0,1)  #por filas

##Matrix = ithaca['log P'].as_matrix()
##ithaca['log P'].reset_index().values
##lnP=log(P)
##ithaca.ix[1940:1950,:] # con .ix son posiciones de 0 a n-1?
##ithaca.iloc[1940:1950,:] # con .iloc son longitudes de 1 a n pero admite 0?
##df.ix[1] == df.iloc[1] pero df.ix[1:3] == df.iloc[1:2]?
lnP=ithaca['log P'].values
##ithaca_resample = bootstrap_resample(lnP, n=10000)
##print 'original mean:', ithaca['log P'].mean()
##print 'resampled mean:', ithaca_resample.mean()
##print 'original std:', ithaca['log P'].std(ddof=1)
##print 'resampled std:', ithaca_resample.std(ddof=1)
##print '-' * 50
##ithaca_resampled2 = pd.DataFrame(index=ithaca.index, columns=ithaca.columns,
##                                 dtype=ithaca.dtypes)
##for col in ithaca.columns:
##    ithaca_resampled2[col] = bootstrap_resample2(ithaca[col])
##comparar = pd.DataFrame([(ithaca.mean()['log P'],
##                          ithaca_resampled2.mean()['log P']),
##                         (ithaca.std(ddof=1)['log P'],
##                          ithaca_resampled2.std(ddof=1)['log P'])],
##                        index=['media lnx', 'std lnx'],
##                        columns=['original', 'Bootstrap'])
##comparar['Diferencia'] = comparar['original'] - comparar['Bootstrap']
##print 'Comparativa de la desviacion estandar de log P'
##print comparar

nb=1000
a=0.05
ithaca_resampled2 = pd.DataFrame(columns=ithaca.columns, dtype=ithaca.dtypes)
for col in ithaca.columns:
    ithaca_resampled2[col] = bootstrap_resample2(ithaca[col],n=nb)
##ithaca_resample = bootstrap_resample(lnP, n=10000)
##print '+Original mean:'
##print ithaca.mean()
##print '+Resampled mean:'
##print ithaca_resampled2.mean()
##print '+Original std:'
##print ithaca.std(ddof=1)
##print '+Resampled std:'
##print ithaca_resampled2.std(ddof=1)
##print '-' * 50
comparar = pd.DataFrame([ithaca.mean().values, ithaca_resampled2.mean().values,
                         ithaca.std(ddof=1).values,
                         ithaca_resampled2.std(ddof=1).values],
                        index=['media','media resampled', 'std','std resampled'],
                        columns=ithaca.columns).convert_objects(convert_numeric=True)
print comparar


icm = icnorm_media(ithaca, a)
print icm
icv = icnorm_var(ithaca, a)
print icv


##pd.DataFrame(columns=ithaca.columns, copy=True)
##icm2 = icnorm_media(ithaca_resampled2, a)
##print icm2
##icv2 = icnorm_var(ithaca_resampled2, a)
##print icv2

def quantile2(x, q):
    x_i=pd.DataFrame(index=[q],
                     columns=ithaca.columns).convert_objects(convert_numeric=True)
    for i in range(len(q)):
        x_i.loc[q[i]] = ithaca.quantile(q[i]).values
    return x_i

percentiles = quantile2(ithaca_resampled2,[1-a/2, a/2])
print percentiles

##################################################################################
def bootstrap_resample1(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        filas, columnas = X.index, X.columns
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)       
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X.irow(resample_i))
    X_resample = pd.DataFrame(X_resample, index=filas, columns=columnas)
    return X_resample


def bootstrap_resampleB(x, n=1000):
    """ Bootstrap resample repeat n times
    """
    x_i=pd.DataFrame(index=[range(n)], columns=ithaca.columns)    
    for i in range(n):
        x_resample = bootstrap_resample1(x)
        x_i.loc[i] = x_resample.std(ddof=1).values
##        x_i.astype(x.loc[i].dtypes)
    return x_i.convert_objects(convert_numeric=True)

ithaca_std = bootstrap_resampleB(ithaca)
ithaca_std = pd.DataFrame(ithaca_std, dtype=ithaca.dtypes)
##print ithaca_std
percentiles = quantile2(ithaca_std,[a/2, 1-a/2])
print percentiles

ithaca_std_sort = pd.DataFrame(ithaca_std['log P'], copy=True)
ithaca_std_sort = ithaca_std_sort.sort('log P')
print ithaca_std_sort.ix[1000*(1-0.05/2)-1]
print ithaca_std_sort.ix[1000*0.05/2-1]



