import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.utils import shuffle
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway

#Function for importing SS dataset.
def import_SS(normalize):
    print('Loading SS dataset...')
    dataset = pd.read_excel('../../source/SkinSegmentation/SkinSegmentation.xlsx')
    #dataset = pd.read_excel('../../source/SkinSegmentation/SkinSegmentation_1000.xlsx')
    dataset = shuffle(dataset)
    dataset.reset_index(inplace=True, drop=True)
    if normalize:
        dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
    print('Done!')
    return dataset

# Function for returning a random row of a dataset.
def random_row(dataset):
    index = random.randint(0, len(dataset.index) - 1)
    return dataset.iloc[index,:]

# Define method for finding a prediction given x and theta vectors.
def get_predictions(x, coeffs):
    predictions = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        predictions[i] = p(x[i], coeffs)

    return predictions

# Define method for calculating p(x;theta), the log-odds function.
def p(x, coeffs):
    power = np.dot(x, coeffs)
    #print(power)
    return (1 / (1 + np.exp(-power)))

# Define method for calculating log-likelihood.
def log_likelihood(y, x, coeffs):
    ans = 0
    predictions = np.zeros(len(x))
    for i in range(len(x)):
        prediction = -np.dot(x[i], coeffs)
        #print(prediction, np.log(1 / (1 + np.exp(prediction))))
        ans += y[i]*prediction - np.log(1 / (1 + np.exp(prediction)))
    #return np.sum(y*predictions - np.log(2 + np.exp(predictions)))
    return ans

# Define method for calculating the gradient.
def gradient(y, x, coeffs):
    predictions = get_predictions(x, coeffs)
    grad = np.dot(x.T, (predictions - y)) / float(len(y))
    return grad

# Define method for calculating the Hessian matrix.
def hessian_matrix(x, coeffs):
    n = x.shape[1]
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Iterate over all samples.
            for k in range(x.shape[0]):
                H[i][j] += hessian_element(x[k].A1, coeffs, i, j)

    return H / x.shape[0]

# Define method for calculating a single element of the Hessian, given indices i and j.
def hessian_element(x, coeffs, i, j):
    a, b = 1, 1

    # Store the coefficients.
    if i > 0:
        a = x[i]
    if j > 0:
        b = x[j]

    return a * b * (p(x, coeffs) * (1 - p(x, coeffs)))
    #return a * b * partial_2(x, coeffs, 5000)

# Define method for getting the samples of data from a dataset.
def get_samples(x_train, y_train):
    #x, y = np.zeros(x_train.shape[0], x_train.shape[1] + 1), np.zeros(y_train.shape)
    x, y = [], []

    for index, data in x_train.iterrows():
        row = [1.0, float(data['B']), float(data['G']), float(data['R'])]
        x.append(row)

    for index, data in y_train.iteritems():
        y.append(data - 1)

    return np.matrix(x), np.array(y)

# Define method for loss.
def loss(y_real, y_pred):
    return (np.mean(-y_real * np.log(y_pred) - (1 - y_real) * np.log(1 - y_pred)))

# Definition for plotting loss values, for two-dimensional case.
def plot_loss(epochs, losses, dataset, title, color):
    x_axis = np.linspace(0, epochs, len(losses), endpoint=True)
    plt.semilogy(x_axis, losses, color=color)

    # Line at loss = std(y)^2.
    y_real = dataset.iloc[:,4].values
    plt.axhline(y=np.std(y_real)**2, color='darkslategray', xmin=0, xmax=epochs, linestyle='-')

    loss_patch = mpatches.Patch(color=color)
    std_patch = mpatches.Patch(color='darkslategray')
    plt.legend([loss_patch, std_patch], ['MSE', 'Standard Deviation'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(title)
    plt.show()

# Definition for plotting loss values, for two-dimensional case.
def plot_test(losses, y_test, title, color):
    x_axis = np.linspace(0, len(losses), len(losses), endpoint=True)
    plt.semilogy(x_axis, losses, color=color)

    # Line at loss = std(y)^2.
    plt.axhline(y=np.std(y_test)**2, color='darkslategray', xmin=0, xmax=len(losses), linestyle='-')

    loss_patch = mpatches.Patch(color=color)
    std_patch = mpatches.Patch(color='darkslategray')
    plt.legend([loss_patch, std_patch], ['Loss', 'Standard Deviation'])
    plt.xlabel('Value')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
