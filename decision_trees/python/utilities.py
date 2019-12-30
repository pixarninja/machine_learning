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

#Function for importing CCPP dataset.
def import_CCPP(normalize):
    print('Loading CCPP dataset...')
    dataset = pd.read_excel('../../source/CCPP/Folds5x2_pp.xlsx')
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

# Define method for classification of a stump given a row.
def evaluate(stumps, row):
    val = 0
    alpha = 0.5
    for stump in stumps:
        if(row[stump['attribute']] <= stump['value']):
            val += alpha * stump['left']
        else:
            val += alpha * stump['right']
    return val

# Define method for calculating mean squared error.
def mse(dataset, stumps):
    error = 0
    for index, data in dataset.iterrows():
        error += (data['PE'] - evaluate(stumps, data))**2
    return error / len(dataset.index)

# Define loss function.
def loss(a, b):
    return (a - b) ** 2

# Definition for approximating the gradient of the objective function.
def gradient(dataset, stumps):
    y = dataset.iloc[:,4].values
    if len(stumps) == 0:
        return (np.sum(y) / len(y))
    else:
        error = 0
        for index, data in dataset.iterrows():
            #error += val - evaluate(stumps[-1], dataset.iloc[i,:])
            error +=  y[index] - evaluate(stumps[-1], data)
        return error / len(y)

# Definition for Squared Sum of Errors.
def SSE(dataset, stumps):
    sse = 0
    for index, data in dataset.iterrows():
        y_real = data['PE']
        y_pred = evaluate(stumps, data)
        sse += (y_real - y_pred) ** 2

    return sse

# Definition for Squared Sum of Total Error.
def SST(dataset):
    sst = 0
    y_mean = np.mean(dataset.iloc[:,4].values)
    for index, data in dataset.iterrows():
        y_real = data['PE']

        sst += (y_real - y_mean) ** 2

    return sst

# Definition for R^2.
def R_squared(dataset, stumps):
    return (1 - SSE(dataset, stumps) / SST(dataset))

def T_test(dataset, stumps):
    y_real = np.array(dataset.iloc[:,4].values)
    y_pred = []
    for index, data in dataset.iterrows():
        y_pred.append(evaluate(stumps, data))

    y_pred = np.array(y_pred)

    print(ttest_ind(y_pred, y_real))
    print(ttest_ind(y_pred, y_real, equal_var=False))
    print(ttest_rel(y_pred, y_real))
    print(f_oneway(y_pred, y_real))

# Definition for overall statistics function.
def stats(dataset, stumps, title):
    print('Stats For ' + title + ':')
    T_test(dataset, stumps)
    print('R-squared: ', R_squared(dataset, stumps))
    print('\n')

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
