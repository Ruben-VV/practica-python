# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab, mlab, pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from numpy import *
from math import factorial
import scipy.misc
from scipy.stats import *
import random

def bootstrap2(data, num_samples=10000, alpha=0.05, statistic=None, replace=True):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    if replace==True:
        idx = np.random.randint(0, n, (num_samples, n))
        samples = data[idx]
    else:
        samples=random.sample(data,n)   
        for i in xrange(num_samples-1):
            samples=np.append(samples,random.sample(data,n))
        samples=reshape(samples,(num_samples, n))    
    
    if statistic == None: #
        statistic = [np.mean, np.std] #
        
    if np.size(statistic) == 1:
        if statistic == np.std:
            stat = np.sort(statistic(samples, 1,ddof=1))
        else:
            stat = np.sort(statistic(samples, 1))
            
        ci = [stat[int((alpha/2.0)*num_samples)],
                stat[int((1-alpha/2.0)*num_samples)]]
        return (stat, ci)
    elif np.size(statistic) == 2:
        stat_mean = np.sort(statistic[0](samples, 1))
        stat_std = np.sort(statistic[1](samples, 1, ddof=1))  
        ci_mean = [stat_mean[int((alpha/2.0)*num_samples)], stat_mean[int((1-alpha/2.0)*num_samples)]]
        ci_std = [stat_std[int((alpha/2.0)*num_samples)], stat_std[int((1-alpha/2.0)*num_samples)]]
        return ( stat_mean, ci_mean, stat_std, ci_std)
    else:
        print 'Too many arguments in statistic'



def plot_bootstrap2(stat, ci_stat, statistic="Mean"):
    #pylab.figure(figsize=(8,4))
    pylab.hist(stat, 100, histtype='step')
    color = ['red', 'green']
    for i in xrange(2):
        pylab.axvline(ci_stat[i], c=color[i])
    pylab.title("Historgram of data for "+ statistic + "\'s Bootstrap")
    print "Bootstrapped 95% confidence intervals of "+ statistic + "\nLow:", ci_stat[0],\
          "\nHigh:", ci_stat[1]
    print "--"*25
    

if __name__ == "__main__":
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
    ithaca['log P'] = np.log(ithaca['P'])
    x = ithaca['log P'].values

    stat_mean, ci_mean = bootstrap2(x,statistic=np.mean)
    obs_mean = np.mean(x)
    print "Mean of sample data: \n", obs_mean
    #pylab.figure(figsize=(8,8))
    pylab.figure()
    pylab.subplot(211)
    plot_bootstrap2(stat_mean, ci_mean)
    pylab.axvline(obs_mean, c='black')

    stat_std, ci_std = bootstrap2(x,statistic=np.std)
    obs_std = np.std(x,ddof=1)
    print "Standard Desviation of sample data: \n", obs_std
    pylab.subplot(212)
    plot_bootstrap2(stat_std, ci_std, "standar desviation")
    pylab.axvline(obs_mean, c='black')
    plt.show()
    
