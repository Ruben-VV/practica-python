def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    import numpy as np, pandas as pd
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


def test_bsr_shape():
    import numpy as np, pandas as pd
    # test without resampling length parameter
    X = np.arange(10000)
    X_resample = bootstrap_resample(X)
    assert X_resample.shape == (10000,), 'resampled length should be 10000'
    
    # test with resampling length parameter
    n = 5000
    X_resample = bootstrap_resample(X, n=n)
    assert X_resample.shape == (n,), 'resampled length should be %d' % n


def test_bsr_mean():
    import numpy as np, pandas as pd
    # test that means are close
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = np.arange(10000)
    X_resample = bootstrap_resample(X, 5000)
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'


def test_bsr_on_df():
    import numpy as np, pandas as pd
    # test that means are close for pd.DataFrame with unusual index
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = pd.Series(np.arange(10000), index=np.arange(10000)*10)
    
    X_resample = bootstrap_resample(X, 5000)
    print X_resample.mean(), X.mean()
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'
    

def test_bsr_on_df():
    import numpy as np, pandas as pd
    # test that means are close for pd.DataFrame with unusual index
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = pd.Series(np.arange(10000), index=np.arange(10000)*10)
        
    X_resample = bootstrap_resample(X, 5000)
    print X_resample.mean(), X.mean()
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'
    
    
def bootstrap_resample2(X, n=None):
    import numpy as np, pandas as pd
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
    return X_resample


def test_bsr_on_df2():
    import numpy as np, pandas as pd
    # test that means are close for pd.DataFrame with unusual index
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = pd.Series(np.arange(10000), index=np.arange(10000)*10)
        
    X_resample = bootstrap_resample(X, 5000)
    print X_resample.mean(), X.mean()
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'
    
    ithaca_resample2 = bootstrap_resample2(ithaca['log P'], 10000)
    print ithaca_resample2.mean(), ithaca['log P'].mean()
    assert abs(ithaca_resample2.mean() - ithaca['log P'].mean()) / ithaca['log P'].mean() < 1e-2, 'means should be approximately equal'
