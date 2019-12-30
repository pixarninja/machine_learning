# Modeled off of "Logistic Regression" on SciKit Learn (Library)
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utilities

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# Load dataset.
dataset = utilities.import_SS(False)
x = dataset.iloc[:,0:3]
y = dataset.iloc[:,3]
offset = int(x.shape[0] * 0.9)
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]

# Fit regression model.
model = LogisticRegression()
fit = model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Print stats
print(model.score(x_test, y_test))
print(model.intercept_, model.coef_)
print(model.n_iter_)
