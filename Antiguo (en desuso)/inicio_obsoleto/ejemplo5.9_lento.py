# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 14:30:16 2015

@author: Ruben
"""

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

from IPython.display import display, Math, Latex

seeded = [49,4,18,26,29,9,16,12,2,22,10,34]
unseeded = [61,33,62,45,0,30,82,10,20,358,63,NaN]
##strikes = pd.DataFrame(array([seeded, unseeded]).T,columns=['seeded', 'unseeded'])
strikes = pd.DataFrame(array([seeded, unseeded])).T
strikes.columns = ['seeded', 'unseeded']


def permutar(xy, nx=None, ny=None, forma=None):
    """ Permutation Test
    xy = [x,y]
    xy = vector con los valores de xy, teniendo primero los de x y luego los de y, su longitud es de nxy
    nxy = nx + ny
    Suponemos nx >= ny
    Realizamos ny veces la permutacion de xy_i con xy_m escogiendo al azar xy_i
    siendo m = nxy, nxy-1, ..., nxy-ny=nx
    
    forma=1 da como salida dos arrays xp, yp
    forma=None or forma!=1 da como salida un array que contiene xp e yp
    
    La salida es un array
    """     
    if len(xy) == 2:
        nx, ny = len(xy[0]), len(xy[1])
        xy = hstack(xy)
        if forma == None:
            forma = 1      
    nxy = len(xy)
    m = range(nxy-1,nx-1,-1)
    # Numero al azar entre [a,b]
    #(b - a) * npr.random() + a
    for i in range(ny):
        pos = int(round(m[i] * npr.random()))
        xy[pos], xy[m[i]] = xy[m[i]], xy[pos]
    if forma == 1:
        return xy[:nx], xy[nx:]
    else:
        return xy
    
if __name__ == '__main__':
    xy0 = [seeded, unseeded[:-1]]
    xyp0 = npr.permutation(hstack(xy0))
    xp0, yp0 = xyp0[:len(seeded)], xyp0[len(seeded):]
    print xp0, yp0
    print len(xp0), len(yp0)
    xp, yp = permutar(xy0)
    print xp, yp
    print len(xp), len(yp)
    #print permutar(hstack(xy0), nx=len(xy0), ny=len(unseeded[:-1]))
    print 'Fin de la pruba\n'
    
def permutarn(datos, n=10000):
    """
    Repetir permutar N veces obteniendo el ratio L-scales
    """
    if len(datos[0]) > len(datos[1]):
        while len(datos[0]) > len(datos[1]):
            datos[1].append(nan)
    elif len(datos[0]) < len(datos[1]):
        while len(datos[0]) > len(datos[1]):
            datos[1].append(nan)
    ratio = zeros(n)
    for i in range(n):        
        xP, yP = permutar(datos)
        Lp = Lscale([xP, yP])
        ratio[i] = Lp[0]/Lp[1]
    print '.'
    return ratio

##Intervalo de un valor x dado
#def get_bin(x, n, bins):
#    for bin in bins:
#        if x < bin:
#            bx = bin
#            pxb = np.where(b == bx)[0][0]
#            pxn = pxb - 1
#            nx = n[pxn]
#            binx = [bins[pxn], bx]
#            break
#    print binx, pxb, pxn, nx
#    return binx, nx
#def max_bin(n, b):
#    bin_max = np.where(n == n.max())[0][0]
#    maxbin = [b[bin_max], b[bin_max+1]]
#    print 'maxbin', ma

if __name__ == '__main__':    
    radio = permutarn(xy0)
    #Histograma de los datos
    pylab.figure()
    n, b, patches = pylab.hist(radio, 100, histtype='step')
    pylab.title('Historgram of data L-scale')
    #get_bin(0.188, n, b)
    #max_bin(n, b)
##############################################################################
##############################################################################
##############################################################################
##############################################################################
def permutacion(datos, n=10000):
    """
    Repetir permutar N veces obteniendo el ratio L-scales
    """
    x, y = array(datos[0]), array(datos[1]) 
    x, y = x[~isnan(x)], y[~isnan(y)]
    nx, ny = len(x), len(y)
    ratio = zeros(n)
    for k in range(n):
        #for i in range(ny):
        datosp = npr.permutation(hstack([x,y]))
        xp, yp = datosp[:nx], datosp[nx:]
            #datos[pos], datos[m[i]] = datos[m[i]], datos[pos]
        Lp = Lscale([xp.tolist(), yp.tolist()])
        ratio[k] = Lp[0]/Lp[1]
    print '.'
    return ratio
  
     
##Intervalo de un valor x dado
#def get_bin2(x, n, bins):
#    for bin in bins:
#        if x < bin:
#            bx = bin
#            pxb = np.where(b == bx)[0]
#            pxn = pxb - 1
#            nx = n[pxn]
#            binx = [bins[pxn], bx]
#            print bx
#            break
#    print binx, pxb, pxn, nx
#    return binx, nx
#def max_bin2(n, b):
#    bin_max = np.where(n == n.max())[0][0]
#    print bin_max
#    maxbin = [b[bin_max], b[bin_max+1]]
#    print maxbin
#    print 'maxbin', maxbin[0], maxbin[1]
#
if __name__ == '__main__':
    radios2 = permutacion([seeded, unseeded])
    #Histograma de los datos
    pylab.figure()
    nper, bper, patchesp = pylab.hist(radios2, 100, histtype='step')
    pylab.title('Historgram of data L-scale')
#    get_bin2(0.188, nper, bper)
#    max_bin2(nper, bper)