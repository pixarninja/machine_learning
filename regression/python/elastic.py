import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import utilities

# Definition of Elastic Net Regression algorithm.
def elastic(dataset):
    # Initialize local variables
    x = dataset.iloc[:,0:4].values
    y = dataset.iloc[:,4].values

    regressor = ElasticNet()
    regressor.fit(x, y) #training the algorithm

    return np.append(regressor.intercept_, regressor.coef_)
