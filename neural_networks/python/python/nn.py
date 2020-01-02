import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import utils as utils

########################
#  NETWORK DEFINITION  #
########################

# Define sigmoid function as activation.
def activation(x):
    return 1 / (1 + np.exp(-x))

# Define softmax function as final layer activation.
def final(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Define initialization step.
def initialize_parameters(shape):
    # Initialize weights for input and hidden layer.
    w = [np.random.randn(shape[1], shape[0]),  # (hidden, input )
         np.random.randn(shape[2], shape[1])]  # (output, hidden)

    # Initialize biases for input and hidden layer.
    b = [np.zeros((shape[1], 1)),              # (hidden, 1     )
         np.zeros((shape[2], 1))]              # (output, 1     )

    # Return the weights and biases.
    return w, b

# Define forward step, return outputs of each layer.
def forward_step(x, w, b):
    res = []
    res.append(activation(np.matmul(w[0], x) + b[0]))
    res.append(final(np.matmul(w[1], res[0]) + b[1]))

    return res

# Define backward step, return calculated gradients.
def backward_step(x, y, outs, w):
    # Initialize result and helper variables.
    res = []
    d2 = outs[1] - y
    d1 = np.multiply(np.dot(w[1].T, d2), 1 - np.power(outs[0], 2))

    res.append([np.dot(d1, x.T) / float(len(x)),
                np.sum(d1, axis=1, keepdims=True) / float(len(x))])

    res.append([np.dot(d2, outs[0].T) / float(len(x)),
                np.sum(d2, axis=1, keepdims=True) / float(len(x))])

    return res

# Define update step.
def gradient_descent(w, b, grads, alpha):
    # Update weights.
    for i, w_i in enumerate(w):
        w[i] = w_i - alpha * grads[i][0]

    # Update biases.
    for i, b_i in enumerate(b):
        b[i] = b_i - alpha * grads[i][1]

    return w, b

########################
#  NETWORK EVALUATION  #
########################

# Define loss calculation for Gradient Descent.
def loss(y_real, y_pred):
    return -np.sum(np.multiply(y_real, np.log(y_pred))) / float(y_real.shape[1])

# Define accuracy calculation.
def accuracy(y_real, y_pred):
    s = 0
    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            s += 1

    return s / float(len(y_real))

# Define confusion matrix calculation.
def confusion(y_real, y_pred):
    predictions = np.argmax(y_pred, axis=0)
    labels = np.argmax(y_real, axis=0)

    print(confusion_matrix(predictions, labels))
    print(classification_report(predictions, labels))

########################
########  MAIN  ########
########################

# Define how to run the dense Neural Network.
def neural_network(dataset, num_epochs, net_hid, alpha):
    x_train, y_train, x_test, y_test = [], [], [], []
    net_inp, net_out = 1, 1
    losses = []

    if dataset == 'MNIST':
        # Import training and testing data from TensorFlow API.
        x_train, y_train, x_test, y_test = utils.import_MNIST()
        print(x_train.shape, y_train.shape)
        net_inp = x_train.shape[0]
        net_out = 10
    elif dataset == 'CIFAR-10':
        # Import training and testing data from files.
        x_train, y_train, x_test, y_test = utils.import_CIFAR(10)
        print(x_train.shape, y_train.shape)
        net_inp = x_train.shape[0]
        net_out = 10
    elif dataset == 'CIFAR-100':
        # Import training and testing data from files.
        x_train, y_train, x_test, y_test = utils.import_CIFAR(100)
        print(x_train.shape, y_train.shape)
        net_inp = x_train.shape[0]
        net_out = 100
    else:
        print('Please enter a valid dataset [MNIST, CIFAR-10, CIFAR-100]')
        return

    net_shape = (net_inp, net_hid, net_out)

    # Initialize network weights and biases.
    w, b = initialize_parameters(net_shape)

    for i in range(num_epochs):
        res = forward_step(x_train, w, b)

        grads = backward_step(x_train, y_train, res, w)
        w, b = gradient_descent(w, b, grads, alpha)

        losses.append(loss(y_train, res[1]))
        if (i % 100 == 0):
            print('{0}: {1}'.format(i, losses[i]))

    # Create final outputs.
    res = forward_step(x_test, w, b)
    confusion(y_test, res[1])

    return losses

#num_epochs = 5000
#hidden_layers = 64
#learning_rate = 0.0001
#neural_network('CIFAR-10', num_epochs, hidden_layers, learning_rate)
