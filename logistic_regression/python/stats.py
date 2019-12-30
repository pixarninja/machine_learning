import math as math
import pandas as pd
import numpy as np
import utilities
import sgd
import newton

# Define method for evaluating the fit of a model.
def evaluate_model(x, y, coeffs):
    correct = 0
    incorrect = 0
    predictions = utilities.get_predictions(x, coeffs)

    for i in range(len(predictions)):
        # Correct prediction.
        if (y[i] == 0 and predictions[i] < 0.5) or (y[i] == 1 and predictions[i] >= 0.5):
            correct += 1
        # Incorrect prediction.
        else:
            incorrect += 1

    print('Accuracy: %0.06f' % (correct / float(correct + incorrect)))

# Import dataset.
dataset = utilities.import_SS(False)

# Split dataset into training and test data.
x_data = dataset.iloc[:,0:3]
y_data = dataset.iloc[:,3]
offset = int(x_data.shape[0] * 0.1)
x_train, y_train = x_data[:offset], y_data[:offset]
x_test, y_test = x_data[offset:], y_data[offset:]

# Get samples.
x, y = utilities.get_samples(x_train, y_train)

# Format:
# *.model(X, y, max_epochs, epsilon, beta (learning parameter))

#coeffs = sgd.model(x, y, 100, 0.0001, 0.001)
#print('SGD Stats...')
#print(coeffs)
#x, y = utilities.get_samples(x_test, y_test)
#evaluate_model(x, y, coeffs)

coeffs = newton.model(x, y, 100, 0.0001, 0.01)
print('Newton Stats:')
print(coeffs)
x, y = utilities.get_samples(x_test, y_test)
evaluate_model(x, y, coeffs)
