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

import pandas as pd
import IC
from IC import *


seeded = [49,4,18,26,29,9,16,12,2,22,10,34]
unseeded = [61,33,62,45,0,30,82,10,20,358,63,NaN]
strikes = pd.DataFrame(seeded,columns=['seeded'])
strikes['unseeded'] = unseeded
strikes.loc[11,'unseeded']=None

print strikes


##m,ic = icnorm_media(strikes)
icm = icnorm_media(strikes)
##print m
print icm

icv = icnorm_var(strikes)
print icv
