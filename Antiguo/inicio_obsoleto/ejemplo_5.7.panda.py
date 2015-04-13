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

import pandas as pd

seeded = array([49,4,18,26,29,9,16,12,2,22,10,34], ndmin=2).T
unseeded = array([61,33,62,45,0,30,82,10,20,358,63,NaN], ndmin=2).T
S = ['S' for i in range(len(seeded))]
N =  ['N' for i in range(len(unseeded))]
S = chararray((len(seeded), 1))
S[:] = 'S'
N =  chararray((len(unseeded)-1, 1))
N[:] = 'N'

A = column_stack((seeded,unseeded))
##A = vstack((seeded,unseeded)).T        #Si son vectores fila
##A = vstack((seeded,unseeded))
##A = hstack((seeded,unseeded))          #Si son vectores columna
##A = column_stack((seeded,unseeded))
##A[where(~isnan(A))]
B = copy(A)
B[isnan(B)]=-1


media = np.nanmean(A,axis=0)
strikes = np.nansum(A,axis=0)
desviacion = np.nanstd(A,axis=0,ddof=1)
var = np.nanvar(A,axis=0)
print 'Seeded        Unseeded'
print '        Media         '
print str(media[0]) + '         ' + str(round(media[1],2))
print '        Desviacion         '
print str(round(desviacion[0],2)) + '    ' + str(round(desviacion[1],2))


U1, p, R = rangosuma(seeded,unseeded[:11])
R = array([R]).T

sn1 = column_stack((seeded,S))

##SS = reshape(A, (len(A)*2, 1))
SS = vstack((seeded,unseeded[:-1,:]))
SU = vstack((S,N))
dt = [('strikes', '<i4'), ('seed', '|S1'), ('ranks', 'f8')]
SSN = zeros((len(SS), 1), dtype=dt)
SSN['strikes']=SS
SSN['seed']=SU
SSN['ranks']=R
tabla = sort(SSN, 0, order='strikes')
###Datos segregados
##s1 = array([tabla[tabla['seed']=='S']]).T
##s2 = array([tabla[tabla['seed']=='N']]).T
##R1= sum(s1['ranks'])
##R2 = sum(s2['ranks'])

####df.convert_objects(convert_numeric=True)
##df = pd.DataFrame(seeded,columns=['seed'])
####df['S'] = 'S'
##df['S'] = S

df = pd.DataFrame(SSN.flatten())
tablapd = df.sort('strikes')
print tablapd



