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
    dataset = pd.read_excel('./source/CCPP/Folds5x2_pp.xlsx')
    dataset = shuffle(dataset)
    dataset.reset_index(inplace=True, drop=True)
    if normalize:
        dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
    return dataset

# Function for returning a random row of a dataset.
def random_row(dataset):
    index = random.randint(0, len(dataset.index) - 1)
    return dataset.iloc[index,:]

# Define loss function.
def loss(a, b):
    return (a - b) ** 2

# Definition for Squared Sum of Errors.
def SSE(dataset, coeffs):
    sse = 0
    for index, data in dataset.iterrows():
        y_real = data['PE']
        row = [1, data['AT'], data['V'], data['AP'], data['RH']]
        y_pred = np.dot(coeffs, row)

        sse += (y_real - y_pred) ** 2

    return sse

# Definition for Squared Sum of Total Error.
def SST(dataset, coeffs):
    sst = 0
    y_mean = np.mean(dataset.iloc[:,4].values)
    for index, data in dataset.iterrows():
        y_real = data['PE']

        sst += (y_real - y_mean) ** 2

    return sst

# Definition for R^2.
def R_squared(dataset, coeffs):
    return (1 - SSE(dataset, coeffs) / SST(dataset, coeffs))

def T_test(dataset, coeffs):
    y_real = np.array(dataset.iloc[:,4].values)
    y_pred = []
    for index, data in dataset.iterrows():
        row = [1, data['AT'], data['V'], data['AP'], data['RH']]
        pe_pred = np.dot(coeffs, row)
        y_pred.append(pe_pred)

    y_pred = np.array(y_pred)

    print(ttest_ind(y_pred, y_real))
    print(ttest_ind(y_pred, y_real, equal_var=False))
    print(ttest_rel(y_pred, y_real))
    print(f_oneway(y_pred, y_real))

# Definition for overall statistics function.
def stats(dataset, fit, title):
    print('Stats For ' + title + ':')
    T_test(dataset, fit)
    print('Fit:       ', fit)
    print('R-squared: ', R_squared(dataset, fit))
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
    plt.legend([loss_patch, std_patch], ['Loss', 'Standard Deviation'])
    plt.xlabel('Epoch')
    plt.ylabel('Logarithmic Loss')
    plt.title(title)
    plt.show()

# Definition for plotting loss values, for two-dimensional case.
def plot_fit(dataset, coeffs, title, color):
    losses = []
    y_real = np.array(dataset.iloc[:,4].values)
    for index, data in dataset.iterrows():
        row = [1, data['AT'], data['V'], data['AP'], data['RH']]
        y_pred = np.dot(coeffs, row)
        losses.append(loss(y_real[index], y_pred))

    x_axis = np.linspace(0, len(dataset.iloc[:,4].values), len(losses), endpoint=True)
    plt.plot(x_axis, losses, color=color)
    plt.xlim(0, len(dataset.iloc[:,4].values))

    # Line at loss = std(y)^2.
    plt.axhline(y=np.std(y_real)**2, color='darkslategray', xmin=0, xmax=len(dataset.iloc[:,4].values), linestyle='-')

    loss_patch = mpatches.Patch(color=color)
    std_patch = mpatches.Patch(color='darkslategray')
    plt.legend([loss_patch, std_patch], ['Loss', 'Standard Deviation'])
    plt.xlabel('Value')
    plt.ylabel('Logarithmic Loss')
    plt.title(title)
    plt.show()
