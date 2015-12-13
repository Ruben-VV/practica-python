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


def bootstrap(data, num_samples, alpha, statistic=None):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    if statistic == None: #
        statistic = [np.mean, np.std] #
        
    if size(statistic) == 1:
        if statistic == np.std or statistic == std:
            stat = np.sort(statistic(samples, 1,ddof=1))
        else:
            stat = np.sort(statistic(samples, 1))
        return (stat[int((alpha/2.0)*num_samples)],
                stat[int((1-alpha/2.0)*num_samples)],stat)
    elif size(statistic) == 2:
        stat_mean = np.sort(statistic[0](samples, 1))
        stat_std = np.sort(statistic[1](samples, 1, ddof=1))  
        return (stat_mean[int((alpha/2.0)*num_samples)], stat_mean[int((1-alpha/2.0)*num_samples)], stat_mean, 
                stat_std[int((alpha/2.0)*num_samples)], stat_std[int((1-alpha/2.0)*num_samples)], stat_std)
    else:
        print 'Too many arguments in statistic'
        
