import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import utilities

# Definition of Least Squares Linear Regression algorithm.
def lslr(dataset):
    # Initialize local variables
    x = dataset.iloc[:,0:4].values
    y = dataset.iloc[:,4].values

    regressor = LinearRegression()
    regressor.fit(x, y) #training the algorithm

    return np.append(regressor.intercept_, regressor.coef_)
