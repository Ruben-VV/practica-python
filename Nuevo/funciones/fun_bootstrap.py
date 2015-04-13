# -*- coding: utf-8 -*-
"""
Bootstrap
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab, mlab, pyplot

def bootstrap(data, statistic=None, n_bootstraps=10000, seed=None, ci_width=0.95):
    if statistic==None:
        statistic = ["mean", "std"]
    else:
        statistic = [statistic]
    data = np.asarray(data)
    rng = np.random.RandomState(seed)
    n = len(data)
    
    idx = rng.random_integers(0, n-1, (n_bootstraps, n))
    samples = data[idx]
    
    bootstrapped = []
    for fun in statistic:
        arg=""
        if fun=="std":
            arg = ",ddof=1"
        bootstrapped.append(eval("np."+fun+"(samples,1"+arg+")"))        
        [x.sort() for x in bootstrapped]
    
    ci_min = (1 - ci_width) / 2.
    ci_max = 1. - ci_min
    ci = [[x[int(ci_min * n_bootstraps)], x[int(ci_max * n_bootstraps)]] for x in bootstrapped]

    return bootstrapped, ci    
    

def plot_bootstrap(stat, ci_stat, statistic="Mean"):    
    pylab.figure(figsize=(8,4))
    pylab.hist(stat[0], 100, histtype='step')
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

    stat_mean, ci_mean = bootstrap(x,"mean")
    obs_mean = np.mean(x)
    print "Mean of sample data: \n", obs_mean
    plot_bootstrap(stat_mean, ci_mean[0])
    pylab.axvline(obs_mean, c='black')

    stat_std, ci_std = bootstrap(x,"std")
    obs_std = np.std(x,ddof=1)
    print "Standard Desviation of sample data: \n", obs_std
    plot_bootstrap(stat_std, ci_std[0], "STD")
    pylab.axvline(obs_mean, c='black')
    plt.show()
    
    
