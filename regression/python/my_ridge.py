import pandas as pd
import numpy as np
import random
import utilities
from sgd import sgd

# Definition of Ridge Regression algorithm.
def my_ridge(dataset, max_epochs, alpha, beta):
    # Initialize local variables
    coeffs = [0.0 for i in range(len(dataset.iloc[0,:]))]
    losses = []
    epochs = 0

    # Iterate over the dataset until max epochs has been reached.
    while epochs < max_epochs:
        for index, data in dataset.iterrows():
            # Run the SGD algorithm.
            coeffs, y_real, y_pred = sgd(data, coeffs, alpha, beta)

            # Record loss for this epoch.
            losses.append(utilities.loss(y_real, y_pred))

            # Stop conditions.
            epochs += 1
            if epochs >= max_epochs:
                break

    print(epochs)
    return coeffs, losses
