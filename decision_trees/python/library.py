# Modeled off of "Gradient Boosting regression" on SciKit Learn (Library)
# Source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utilities

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

# Definition for Squared Sum of Errors.
def SSE(y_real, y_pred):
    sse = 0
    for index in range(len(y_real)):
        sse += (y_real[index] - y_pred[index]) ** 2

    return sse

# Definition for Squared Sum of Total Error.
def SST(y_real):
    sst = 0
    y_mean = np.mean(y_real)
    for index in range(len(y_real)):
        sst += (y_real[index] - y_mean) ** 2

    return sst

# Definition for R^2.
def R_squared(y_real, y_pred):
    return (1 - SSE(y_real, y_pred) / SST(y_real))

# Load dataset.
dataset = utilities.import_CCPP(False)
x = dataset.iloc[:,0:4]
y = dataset.iloc[:,4]
offset = int(x.shape[0] * 0.9)
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]

# Fit regression model.
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x_train, y_train)
mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %.4f" % mse)

losses = (y_test.values.tolist() - clf.predict(x_test)) ** 2

# Plotting.
utilities.plot_test(losses, y_test.values.tolist(), 'Python-Library Loss for Test Dataset', 'palevioletred')
print(R_squared(y_test.values.tolist(), clf.predict(x_test).tolist()))
