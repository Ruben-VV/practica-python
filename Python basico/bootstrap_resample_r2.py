import numpy as np
import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot
import numpy.random as npr

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *
##from math import factorial
from scipy.stats import *

import pandas as pd

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

def bootstrap2(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    if statistic == np.std or statistic == std:
        stat = np.sort(statistic(samples, 1,ddof=1))
    else:
        stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)],stat)
if __name__ == '__main__':
    # data of interest is bimodal and obviously not normal
    # x = np.concatenate([npr.normal(3, 1, 100), npr.normal(6, 2, 200)])
    #Jannuary precipitation at Ithaca, NY, 1933-1982, inches
    P = [0.44, 1.18, 2.69, 2.08, 3.66, 1.72, 2.82, 0.72, 1.46, 1.30, 1.35, .54,
         2.74, 1.13, 2.5, 1.72, 2.27, 2.82, 1.98, 2.44, 2.53, 2.0, 1.12, 2.13,
         1.36, 4.9, 2.94, 1.75, 1.69, 1.88, 1.31, 1.76, 2.17, 2.38, 1.16, 1.39,
         1.36, 1.03, 1.11, 1.35, 1.44, 1.84, 1.69,
         3.0, 1.36, 6.37, 4.55, .52, .87, 1.51]
    year = range(1933,1982+1,1)
    ithaca = pd.DataFrame(P,columns=['P'],index=year)
    ithaca['log P'] = log(ithaca['P'])
    slnx = std(ithaca['log P'],1)
    lnP=ithaca['log P'].values
    
    x = ithaca['log P'].values
    # find mean 95% CI and 100,000 bootstrap samples
    low, high, stat = bootstrap2(x, 100000, np.mean, 0.05)
    # make plots
    pylab.figure(figsize=(12,8))
    pylab.subplot(221)
    pylab.hist(stat, 50, histtype='step')
    pylab.title('Historgram of data for mean')
    pylab.subplot(222)
    pylab.plot([-0.03,0.03], [np.mean(x), np.mean(x)], 'r', linewidth=2)
    pylab.scatter(0.1*(npr.random(len(x))-0.5), x)
    pylab.plot([0.1,0.2], [low, low], 'r', linewidth=2)
    pylab.plot([0.1,0.2], [high, high], 'r', linewidth=2)
    pylab.plot([0.15,0.15], [low, high], 'r', linewidth=2)
    pylab.xlim([-0.2, 0.3])
    pylab.title('Bootstrap 95% CI for mean')
    #pylab.savefig('examples/boostrap.png')
    print "Bootstrapped 95% confidence intervals of Mean\nLow:", low, "\nHigh:", high    
    
    # find mean 95% CI and 100,000 bootstrap samples
    low, high, stat = bootstrap2(x, 100000, np.std, 0.05) 
    # make plots    
    pylab.subplot(223)
    pylab.hist(stat, 50, histtype='step')
    pylab.title('Historgram of data for std')
    pylab.subplot(224)
    pylab.plot([-0.03,0.03], [np.std(x,ddof=1), np.std(x,ddof=1)], 'r', linewidth=2)
    pylab.scatter(0.1*(npr.random(len(x))-0.5), x)
    pylab.plot([0.1,0.2], [low, low], 'r', linewidth=2)
    pylab.plot([0.1,0.2], [high, high], 'r', linewidth=2)
    pylab.plot([0.15,0.15], [low, high], 'r', linewidth=2)
    pylab.xlim([-0.2, 0.3])
    pylab.title('Bootstrap 95% CI for std')
    #pylab.savefig('examples/boostrap.png')
    print "Bootstrapped 95% confidence intervals of Standard Deviation\nLow:", low, "\nHigh:", high
    show()
