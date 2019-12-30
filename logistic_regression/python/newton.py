import math as math
import pandas as pd
import numpy as np
import utilities

# Define method for calculating Newton-Raphson value(s).
def model(x, y, max_epochs, epsilon, beta):
    # Initialize variables.
    coeffs = np.zeros(x.shape[1])
    l_delta = epsilon + 1
    l_curr = 0
    epochs = 0

    while epochs < max_epochs and l_delta > epsilon:
        # Find the gradient and Hessian using current coefficients.
        grad = utilities.gradient(y, x, coeffs)
        hess = utilities.hessian_matrix(x, coeffs)
        try:
            hess = np.linalg.inv(hess)
        except:
            hess = np.identity(len(x[0]), dtype=float)

        # Update step.
        step = np.dot(hess, grad.T)
        print('Update Step: ', step.A1)
        for i in range(len(coeffs)):
            if epochs == 0:
                coeffs[i] = step[i]
            else:
                coeffs[i] = coeffs[i] - beta * step[i]

        predictions = utilities.get_predictions(x, coeffs)
        l_calc = utilities.loss(y, predictions)
        l_delta = np.abs(l_calc - l_curr)
        l_curr = l_calc
        epochs += 1

        print('Coeffs: ', coeffs)
        print('%d.\tLoss: %0.06f' % (epochs, l_calc))
        print('Loss Deltas: ', l_delta)

    return coeffs
