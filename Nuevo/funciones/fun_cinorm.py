import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2#, sem
#import scipy.misc


def ci_norm(x, a=0.05, sigma=None):
    """Calculo de Intervalos de Confianza para una distribucion normal
    IC para distribucion normal, seleccionando casos dependiendo si conocemos
    la desviacion poblacional sigma (o la varianza var = sigma**2)
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
        if isinstance(x,pd.DataFrame):
            m = x.mean().values
            s = x.std(ddof=1).values
            s2 = x.var(ddof=1).values
        else:
            m = x.mean(axis=0)
            s = x.std(axis=0,ddof=1)
            s2 = x.var(axis=0,ddof=1)
        if n > 30:
            z = -norm.ppf(a/2.)
        else:
            z = -t.ppf(a/2.,n-1)
##            z = t.ppf((1+c)/2,n-1)
##scipy.stats.sem(x) calcula el error estandar que es igual a std/sqrt(n)
##    m, se = np.mean(x), scipy.stats.sem(x)
##    h = se * z
##    se = sem(x)
##    h2 = se * z
    z1, z2 = chi2.ppf(1-a/2,n-1), chi2.ppf(a/2,n-1)
    i1, i2 = (n-1) * s2 / z1, (n-1) * s2 / z2
    h = s * z / np.sqrt(n)
    
    data = [m, m-h, m+h, s2, i1, i2, s, np.sqrt(i1), np.sqrt(i2)]
    #index_mean = ['Mean', 'CI_low', 'CI_high']
    index = ['MEAN', 'low_mean', 'high_mean', 'VARIANCE', 'low_var', 'high_var',\
             'STD', 'low_std', 'high_std']

    if isinstance(x, pd.DataFrame):
        ci = pd.DataFrame(data, index=index, columns=x.columns)
    else:
        ci = pd.DataFrame(data, index=index, columns=[str((1-a)*100) + '%'])
    return ci


if __name__ == "__main__":
    #Jannuary precipitation at Ithaca, NY, 1933-1982, inches
    P = [0.44, 1.18, 2.69, 2.08, 3.66, 1.72, 2.82, 0.72, 1.46, 1.30, 1.35, .54,
         2.74, 1.13, 2.5, 1.72, 2.27, 2.82, 1.98, 2.44, 2.53, 2.0, 1.12, 2.13,
         1.36, 4.9, 2.94, 1.75, 1.69, 1.88, 1.31, 1.76, 2.17, 2.38, 1.16, 1.39,
         1.36, 1.03, 1.11, 1.35, 1.44, 1.84, 1.69,
         3.0, 1.36, 6.37, 4.55, .52, .87, 1.51]
    year = range(1933,1982+1,1)
    ithaca = pd.DataFrame(P,columns=['P'],index=year)
    ithaca['log P'] = np.log(ithaca['P'])
    #x = ithaca['log P'].values
    print ci_norm(ithaca)
    print "*" * 30
    
