import pandas as pd
import numpy as np

# Definition of Stochastic Gradient Descent algorithm.
def sgd(data, coeffs, alpha, beta):
    # Setup y and y_hat.
    row = [1, data['AT'], data['V'], data['AP'], data['RH']]
    y_real = data['PE']
    y_pred = np.dot(row, coeffs)

    # Calculate next values.
    for i in range(0, len(row)):
        coeffs[i] = coeffs[i] - alpha * gradient(i, row[i], y_real, y_pred, beta)

    return coeffs, y_real, y_pred

# Definition for approximating the gradient of the objective function.
def gradient(i, x, y_real, y_pred, alpha):
    if i > 0:
        return (-2 * x * (y_real - y_pred)) + (2 * alpha * x)
    else:
        return (-2 * (y_real - y_pred)) + (2 * alpha * x)
