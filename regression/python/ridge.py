import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import utilities

# Definition of Ridge Regression algorithm.
def ridge(dataset):
    # Initialize local variables
    x = dataset.iloc[:,0:4].values
    y = dataset.iloc[:,4].values

    regressor = Ridge()
    regressor.fit(x, y) #training the algorithm

    return np.append(regressor.intercept_, regressor.coef_)
