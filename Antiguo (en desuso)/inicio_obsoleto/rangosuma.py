def rangosuma(x, y):
    """
    Computes the Mann-Whitney statistic
    Missing values in `x` and/or `y` are discarded.
    Parameters
    ----------
    x : sequence
    Input
    y : sequence
    Input
    use_continuity : {True, False}, optional
    Whether a continuity correction (1/2.) should be taken into account.
    Returns
    """
    import numpy as np
    import matplotlib
    from matplotlib import pylab, mlab, pyplot
    plt = pyplot
    from IPython.display import display
    from IPython.core.pylabtools import figsize, getfigs

    from matplotlib import pylab, mlab, pyplot    
    import scipy.stats
    from scipy.lib.six import iteritems
    import scipy.special as special
    import numpy.ma as ma
    from numpy import ndarray
    from scipy.stats import mstats
    
    
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    ranks = scipy.stats.rankdata(np.concatenate([x,y]))
    (nx, ny) = (len(x), len(y))
    nt = nx + ny
    U = ranks[:nx].sum() - nx*(nx+1)/2.
    U = max(U, nx*ny - U)
    u = nx*ny - U

    mu = (nx*ny)/2.
    sigsq = (nt**3 - nt)/12.
    ties = mstats.count_tied_groups(ranks)
    sigsq -= np.sum(v*(k**3-k) for (k,v) in iteritems(ties))/12.
    sigsq *= nx*ny/float(nt*(nt-1))

    z = (U - mu) / ma.sqrt(sigsq)
    prob = special.erfc(abs(z)/np.sqrt(2))
    
    return (u, prob, ranks)
