import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import utilities

# Definition of Lasso Regression algorithm.
def lasso(dataset):
    # Initialize local variables
    x = dataset.iloc[:,0:4].values
    y = dataset.iloc[:,4].values

    regressor = Lasso()
    regressor.fit(x, y) #training the algorithm

    return np.append(regressor.intercept_, regressor.coef_)
