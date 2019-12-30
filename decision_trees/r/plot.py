import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Definition for plotting loss values, for two-dimensional case.
def plot_loss(losses, y_test, title, color):
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

losses = np.loadtxt(fname = "./output_losses.txt")
y_test = np.loadtxt(fname = "./output_y_test.txt")
plot_loss(losses, y_test, 'R-Library Loss on Test Dataset', 'goldenrod')
