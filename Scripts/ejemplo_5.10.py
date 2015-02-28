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
from scipy.stats import norm

from rangosuma import *
from bootstrap_test import *


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
ithaca_resample = bootstrap_resample(lnP, n=10000)
print 'original mean:', ithaca['log P'].mean()
print 'resampled mean:', ithaca_resample.mean()
print 'original std:', ithaca['log P'].std(ddof=1)
print 'resampled std:', ithaca_resample.std(ddof=1)
print '-' * 50
ithaca_resampled2 = pd.DataFrame(index=ithaca.index, columns=ithaca.columns,
                                 dtype=ithaca.dtypes)
for col in ithaca.columns:
    ithaca_resampled2[col] = bootstrap_resample2(ithaca[col])
comparar = pd.DataFrame([(ithaca.mean()['log P'],
                          ithaca_resampled2.mean()['log P']),
                         (ithaca.std(ddof=1)['log P'],
                          ithaca_resampled2.std(ddof=1)['log P'])],
                        index=['media lnx', 'std lnx'], columns=['original', 'Bootstrap'])
comparar['Diferencia'] = comparar['original'] - comparar['Bootstrap']


print 'Comparativa de la desviacion estandar de log P'
print comparar



