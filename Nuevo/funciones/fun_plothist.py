def PlotHist(data, log=False, bin=50, cdf="norm", ci=0.95):
    '''Condition: plot histogram of data with n bins and fit to a distribution

    Parameters
    ----------
    data : data for plot a histogram
    log : True for logarithmic histogram
    bin : number of bins or interval of bin. More info from pylab.hist or numpy.histogram
    cdf : cumulative distributive function

    Returns
    -------
    None. Only plot the histogram and print the stadistic D and p value from Kolmogorov-Smirnov Test
    http://www.mathwave.com/help/easyfit/html/analyses/goodness_of_fit/kolmogorov-smirnov.html
    
    '''
    #from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import stats, norm, t
    from numpy import linspace
    from pylab import plot,show,hist
    
    # distribution fitting
    #param = norm.fit(data)
    #mean = param[0]
    #sd = param[1]
    param = eval("scipy.stats."+cdf+".fit(data)")

    #Set large limits
    #xlims = [0, 6*sd+mean]
    xlims = [0.0, 0.8]
    
    #Plot histogram
    histdata = hist(data,bins=bin,alpha=.3,log=log)

    #Generate X points
    x = linspace(xlims[0],xlims[1],500)

    #Get Y points via Normal PDF with fitted parameters
    #pdf_fitted = eval("scipy.stats."+cdf+".pdf(x,loc=mean,scale=sd)")
    #pdf_fitted = norm.pdf(x,loc=mean,scale=sd)

    #Get histogram data, in this case bin edges
    xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in xrange(len(histdata[1])-1)]
    
    #Get bin width from this
    binwidth = (max(xh) - min(xh)) / len(histdata[1])
    scale = len(data)*binwidth
    
    #Scale the fitted PDF by area of the histogram
    #pdf_fitted = pdf_fitted * (len(data) * binwidth)
    
    #params = eval("scipy.stats."+cdf+".fit(data)")
    #fct = eval("scipy.stats."+cdf+".freeze"+str(param))
    fct = eval("scipy.stats."+cdf+".freeze"+str(param))
   
    # Readjust for fct.pdf(x)*scale too high
    xppf = 0
    x0 = fct.ppf(xppf)
    while fct.pdf(x0) > 1.6*histdata[0][0]/scale:
        xppf = xppf + 0.001
        x0 = fct.ppf(xppf)
    x = np.linspace(max((fct.ppf(0.001),fct.ppf(xppf))), fct.ppf(0.999), 500)
    
    #x = np.linspace(fct.ppf(0.001), fct.ppf(0.999), 500)
    
    plt.plot(x, fct.pdf(x)*scale, lw=2, color='red')
    plt.title("Histogram of data and fit distribution '" + cdf + "'")
    #plt.xlim(xmin=0)
    # Only for normal distribution
    if cdf=="norm":
        n=len(x)    
        a = 0.05
        h = -norm.ppf(a/2.) * param[1] #/ np.sqrt(n)
        #print h, param[0]-h, param[0]+h
        plt.axvline(param[0]-h, c='orange')
        plt.axvline(param[0]+h, c='green')
        plt.axvline(param[0], c='black')
    else:
        plt.axvline(data.mean(), c='black')
    #Plot PDF
    #plot(x,pdf_fitted,'r-')
    
    #Kolmogorov–Smirnov test
    D, p = stats.kstest(data, cdf, args=param)
    # Hardcode para la distribucion normal e intervalo de confianza del 95%
    # http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    if cdf=="norm":
        Dc = 1.36/len(data)
        print cdf.ljust(16) + ("p: "+str(p)).ljust(25)+"D: "+str(D)
        print ("D critic (alpha=0.05): ").ljust(25) + str(Dc)
        print "H0: La distribución se ajusta a una distribución " + cdf
        if D > Dc and p<a:
            print "Se rechaza H0 al nivel de significación del "+ str(a*100) + "%"
        else:
            print "Se acepta H0 al nivel de significación del "+ str(a*100) + "%"
    else:
        print cdf.ljust(16) + ("p: "+str(p)).ljust(25)+"D: "+str(D)
        print "H0: La distribución se ajusta a una distribución " + cdf
        if p<a:
            print "Se rechaza H0 al nivel de significación del "+ str(a*100) + "%"
        else:          
            print "Se acepta H0 al nivel de significación del "+ str(a*100) + "%"
    
