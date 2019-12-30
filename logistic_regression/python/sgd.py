import math as math
import pandas as pd
import numpy as np
import utilities

# Definition of Stochastic Gradient Descent algorithm.
def model(x, y, max_epochs, epsilon, beta):
    # Initialize variables.
    coeffs = np.zeros(x.shape[1])
    l_delta = epsilon + 1
    l_curr = 0
    epochs = 0

    while epochs < max_epochs and l_delta > epsilon:
        # Find gradient.
        grad = utilities.gradient(y, x, coeffs)
        print(grad)

        # Update step.
        coeffs -= beta * grad.A1
        predictions = utilities.get_predictions(x, coeffs)
        l_calc = utilities.loss(y, predictions)
        l_delta = np.abs(l_calc - l_curr)
        l_curr = l_calc
        epochs += 1

        print('Coeffs: ', coeffs)
        print('%d.\tLoss: %0.06f' % (epochs, l_calc))
        print('Loss Deltas: ', l_delta)

    return coeffs
