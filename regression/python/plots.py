import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from my_lslr import my_lslr
from my_ridge import my_ridge
from lslr import lslr
from ridge import ridge
from lasso import lasso
from elastic import elastic
import utilities

def plot_evaluation(my_colors, lib_colors):
    # Load dataset.
    dataset = utilities.import_CCPP(True)

    # Store constant variables.
    y_real = np.array(dataset.iloc[:,4].values)
    std_patch = mpatches.Patch(color='darkslategray')
    fits = []
    losses = []
    losses_per_epoch = []

    titles = ['LSLR Loss Per Epoch', 'Ridge Loss Per Epoch']
    patches = ['Our LSLR Loss', 'Our Ridge Loss']
    colors = my_colors
    epochs = 15000

    # Calculate my regression fits.
    fit, loss = my_lslr(dataset, epochs, 0.1)
    fits.append(fit)
    losses_per_epoch.append(loss)
    losses.append(evaluate_fit(dataset, fits[0]))
    fit, loss = my_ridge(dataset, epochs, 0.1, 0.1)
    fits.append(fit)
    losses_per_epoch.append(loss)
    losses.append(evaluate_fit(dataset, fits[1]))

    # Print stats.
    utilities.stats(dataset, fits[0], 'My LSLR')
    utilities.stats(dataset, fits[1], 'My Ridge')

    # Setup plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_axis = np.linspace(0, epochs, len(losses_per_epoch[0]), endpoint=True)
    ax.plot(x_axis, losses_per_epoch[1], color=colors[1], alpha=0.75)
    ax.plot(x_axis, losses_per_epoch[0], color=colors[0], alpha=0.75)
    ax.legend([mpatches.Patch(color=colors[0]), mpatches.Patch(color=colors[1])], [patches[0], patches[1]])
    ax.set(title='Loss Per Epoch', xlabel='Epoch', ylabel='Loss')

    plt.show()

    # Plot fit losses.
    titles = ['Our LSLR', 'Our Ridge']
    x_axis = np.linspace(0, len(dataset.iloc[:,4].values), len(dataset.iloc[:,4].values), endpoint=True)
    fig, ax = plt.subplots(1, 2)

    for i in range(0, len(fits)):
        mean = np.mean(losses[i])
        print(titles[i] + ' Mean: ' + str(mean))
        loss_patch = mpatches.Patch(color=colors[i])
        ax[i].semilogy(x_axis, losses[i], color=colors[i])
        ax[i].set_xlim(0, len(dataset.iloc[:,4].values))
        ax[i].axhline(y=np.std(y_real)**2, color='darkslategray', xmin=0, xmax=len(dataset.iloc[:,4].values), linestyle='-')
        ax[i].axhline(y=mean, color='w', xmin=0, xmax=len(dataset.iloc[:,4].values), linestyle='-')
        ax[i].text(1.05, mean, '{:.4f}'.format(mean), va='center', ha="left", bbox=dict(alpha=0),transform=ax[i].get_yaxis_transform())
        ax[i].legend([loss_patch, std_patch], [patches[i], 'Standard Deviation'])
        ax[i].set(title=titles[i], xlabel='Value', ylabel='Logarithmic Loss')

    plt.subplots_adjust(wspace=0.3)
    plt.show()

    # Reset variables.
    fits = []
    losses = []
    titles = ['LSLR', 'Ridge', 'Lasso', 'Elastic Net']
    patches = ['LSLR Loss', 'Ridge Loss', 'Lasso Loss', 'Elastic Net Loss']
    colors = lib_colors

    # Calculate regression fits from libraries.
    fits.append(lslr(dataset))
    losses.append(evaluate_fit(dataset, fits[0]))
    fits.append(ridge(dataset))
    losses.append(evaluate_fit(dataset, fits[1]))
    fits.append(lasso(dataset))
    losses.append(evaluate_fit(dataset, fits[2]))
    fits.append(elastic(dataset))
    losses.append(evaluate_fit(dataset, fits[3]))

    # Print stats.
    utilities.stats(dataset, fits[0], 'Library LSLR')
    utilities.stats(dataset, fits[1], 'Library Ridge')
    utilities.stats(dataset, fits[2], 'Library Lasso')
    utilities.stats(dataset, fits[3], 'Library Elastic Net')

    # Plot fit losses.
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Evaluation of Regression Libraries')

    for i in range(0, len(fits)):
        mean = np.mean(losses[i])
        print(titles[i] + ' Mean: ' + str(mean))
        loss_patch = mpatches.Patch(color=colors[i])
        ax[int(i / 2) % 2, i % 2].semilogy(x_axis, losses[i], color=colors[i])
        ax[int(i / 2) % 2, i % 2].set_xlim(0, len(dataset.iloc[:,4].values))
        ax[int(i / 2) % 2, i % 2].axhline(y=np.std(y_real)**2, color='darkslategray', xmin=0, xmax=len(dataset.iloc[:,4].values), linestyle='-')
        ax[int(i / 2) % 2, i % 2].axhline(y=mean, color='w', xmin=0, xmax=len(dataset.iloc[:,4].values), linestyle='-')
        ax[int(i / 2) % 2, i % 2].text(1.05, mean, '{:.4f}'.format(mean), va='center', ha="left", bbox=dict(alpha=0),transform=ax[int(i / 2) % 2, i % 2].get_yaxis_transform())
        ax[int(i / 2) % 2, i % 2].legend([loss_patch, std_patch], [patches[i], 'Standard Deviation'])
        ax[int(i / 2) % 2, i % 2].set(title=titles[i], xlabel='Value', ylabel='Loss (Logarithmic)')

    plt.subplots_adjust(wspace=0.3, hspace=0.7)
    plt.show()

# Definition for plotting loss values, for two-dimensional case.
def evaluate_fit(dataset, coeffs):
    losses = []
    y_real = np.array(dataset.iloc[:,4].values)
    for index, data in dataset.iterrows():
        row = [1, data['AT'], data['V'], data['AP'], data['RH']]
        y_pred = np.dot(coeffs, row)
        losses.append(utilities.loss(y_real[index], y_pred))

    return losses

##########
## BODY ##
##########
plot_evaluation(['cornflowerblue', 'palevioletred'], ['lightseagreen', 'indianred', 'darkorchid', 'goldenrod'])
