import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pylab, pyplot
# from matplotlib import mlab
# from numpy import *
# from math import factorial
# import scipy.misc
# from scipy.stats import *
# import scipy.stats


def bootstrap(data, statistic=None, n_B=10000, seed=None, ci_width=0.95):
    '''Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic

    Parameters
    ----------
    data : data for do bootstrap
    statistic : stadistic. Default is None and it does mean and std(ddof=1)
    n_B : number of repeat of bootstrap
    seed : seed for random generation of permutations with repetition
    ci_width: confidance interval inside (0,1)

    Returns
    -------
    estimate : list with resamples of statistic
    ci : Max and Min of Confidance Interval

    Examples
    --------
    '''

    import inspect

    if statistic is None:
        statistic = [np.mean, np.std]
    else:
        assert inspect.isfunction(statistic), \
               "Bad statistic, insert a statistic function"
        statistic = [statistic]
    data = np.asarray(data)
    n = len(data)

    rng = np.random.RandomState(seed)
    idx = rng.random_integers(0, n-1, (n_B, n))
    # idx2 = np.random.randint(0, n, (n_bootstraps, n))
    samples = data[idx]
    # samples2 = data[idx2]

    bootstrapped = []
    for fun in statistic:
        arg = ""
        if fun == np.std:
            arg = ",ddof=1"
#        bootstrapped.append(eval("np."+fun+"(samples,1"+arg+")"))
        bootstrapped.append(eval("np."+fun.__name__+"(samples,1"+arg+")"))
        [bs.sort() for bs in bootstrapped]

    ci_min = (1. - ci_width) / 2.    # ci_min = alpha/2.
    ci_max = 1. - ci_min            # ci_max = 1 - alpha/2.
    ci = [[x[int(ci_min * n_B)], x[int(ci_max * n_B)]] for x in bootstrapped]

    # bootstrapped2 = np.sort(np.mean(samples2,1))
    # ci2 = [bootstrapped2[int(ci_min * n_bootstraps)], bootstrapped2[int(ci_max * n_bootstraps)]]

    if len(statistic) == 1:
        return bootstrapped[0], ci[0]
    else:
        return bootstrapped, ci


def plot_bootstrap(stat, ci_stat, statistic="Mean"):
    '''Plot bootstrap estimate
    '''
#    pylab.figure(figsize=(8,4))
    pylab.hist(stat, 100, histtype='step')
    color = ['red', 'green']
    for i in xrange(2):
        pylab.axvline(ci_stat[i], c=color[i])
    pylab.title("Historgram of data for " + statistic.upper() +
                "\'s Bootstrap")
#    print "Bootstrapped 95% confidence interval of " + statistic + "\nLow:", \
#          ci_stat[0], "\nHigh:", ci_stat[1]
    print "Bootstrapped 95% confidence interval of " + statistic + ": "
    print ci_stat
    print "--"*25


# Example
if __name__ == "__main__":
    import pandas as pd
    # Jannuary precipitation at Ithaca, NY, 1933-1982, inches
    P = [0.44, 1.18, 2.69, 2.08, 3.66, 1.72, 2.82, 0.72, 1.46, 1.30, 1.35, .54,
         2.74, 1.13, 2.5, 1.72, 2.27, 2.82, 1.98, 2.44, 2.53, 2.0, 1.12, 2.13,
         1.36, 4.9, 2.94, 1.75, 1.69, 1.88, 1.31, 1.76, 2.17, 2.38, 1.16, 1.39,
         1.36, 1.03, 1.11, 1.35, 1.44, 1.84, 1.69,
         3.0, 1.36, 6.37, 4.55, .52, .87, 1.51]
    year = range(1933, 1982+1, 1)
    ithaca = pd.DataFrame(P, columns=['P'], index=year)
    # ithaca.dtypes
    ithaca['log P'] = np.log(ithaca['P'])
    x = ithaca['log P'].values

    stadistic = np.mean
    stat_mean, ci_mean = bootstrap(x, np.mean)
    obs_mean = np.mean(x)
    print "Mean of sample data: \n", obs_mean
    pylab.figure(figsize=(8, 8))
    pylab.subplot(211)
    plot_bootstrap(stat_mean, ci_mean)
    pylab.axvline(obs_mean, c='black')

    stat_std, ci_std = bootstrap(x, np.std)
    obs_std = np.std(x, ddof=1)
    print "Standard Desviation of sample data: \n", obs_std
    pylab.subplot(212)
    plot_bootstrap(stat_std, ci_std, "Standar desviation")
    pylab.axvline(obs_mean, c='black')
    pyplot.show()
