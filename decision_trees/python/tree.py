import math as math
import pandas as pd
import numpy as np
import utilities
import sys

# Define method for splitting y on value of x.
def split_col(x, y, val):
    l_set, r_set = [], []
    count = 0
    
    # Add x to l_set or r_set, depending on val.
    #for data in dataset.loc[:,col]
    for index in range(1, len(x)):
        if x[index] <= val:
            l_set.append(y[index])
        else:
            r_set.append(y[index])

    return l_set, r_set

# Define method for splitting dataset on value and column.
def split_dataset(dataset, x, val):
    l_set, r_set = [], []
    
    # Add rows to l_set or r_set, depending on val.
    for index, data in dataset.iterrows():
        if x[index] <= val:
            l_set.append(data.values.tolist())
        else:
            r_set.append(data.values.tolist())

    return np.array(l_set), np.array(r_set)

# Define method for obtaining min gradient for a list of values.
def min_gradient(dataset, x, k):
    # Initialize variables.
    val = 0
    grad = math.inf
    sections = np.array_split(np.sort(x), min(k + 1, len(x)))

    # Use k = len(col = dataset[:,*]), i.e. check each value.
    for i in range(1, len(sections)):
        curr_val = (sections[i][0] + sections[i - 1][len(sections[i - 1]) - 1]) / 2

        # Split dataset based on sample value.
        l_set, r_set = split_dataset(dataset, x, curr_val)
        l_grad = np.average(l_set[:,4]) - np.average(l_set[:,5])
        r_grad = np.average(r_set[:,4]) - np.average(r_set[:,5])
        curr_grad = l_grad + r_grad

        # Store sample if current gradient is lowest.
        if curr_grad < grad:
            val = curr_val
            grad = curr_grad

        sys.stdout.write('\r    [%d/%d][%d] : %f, %f' % (i, k, len(x) - 1, val, grad))

    print('\n')
    return val, grad

# Define method for creating a stump based off an input vector, x.
def create_stump(dataset, name):
    print('\n... Evaluating Column ' + str(name))
    x = dataset[name].values.tolist()
    y = dataset.iloc[:,4].values
    p = dataset.iloc[:,5].values

    # Find split and obtain leaf node values.
    val, grad = min_gradient(dataset, x, 10)
    l_set, r_set = split_col(x, p, val)
    left = np.average(l_set)
    right = np.average(r_set)
    
    epsilon = 0.01
    if abs(left) < epsilon or abs(right) < epsilon:
        print('%s is not applicable!' % (name))
        grad = math.inf

    # Store stump as dictionary of averages.
    stump = {
        'attribute' : name,
        'value' : val,
        'gradient' : grad,
        'left' : left,
        'right' : right
    }

    return stump
    

# Define method for creating the first stump based off variance of the dataset.
def stump_from_dataset(dataset):
    # Intialize variables.
    stump = {}
    var, col, val = 0, 0, 0
    cols = dataset.columns.get_values()
    y = dataset.iloc[:,4].values
    p = dataset.iloc[:,5].values

    # Find which variable to split on, skipping the final y column.
    for i in range(len(cols) - 2):
        # Create stump off of current column.
        curr_stump = create_stump(dataset, cols[i])

        # Test if stump is a better classifier.
        if stump == {} or curr_stump['gradient'] < stump['gradient']:
            stump = curr_stump

    return stump

# Define method for creating another stump (i.e. weak classifier).
def next_stump(dataset, stumps):
    # Initialize helper variables.
    cols = dataset.columns.get_values()
    y = dataset.iloc[:,4].values
    stump = {}
    rim = []

    # ln 1: Special case, average all y(i) values for each x(i).
    #       The feature to use as root is chosen based off max change
    #       in variance.
    if len(stumps) == 0:
        # Overwrite stump value with average.
        init_subset = dataset
        init_subset['PR'] = np.average(y)
        stump = stump_from_dataset(init_subset)

    # Else build off of previous stump using gradient boosting.
    else:
        # ln 2(a): Find r(i)(m) for x(i) in data columns.
        for index, data in dataset.iterrows():
            rim.append(data['PE'] - utilities.evaluate(stumps, data))

        # ln 2(b): Fit a regression tree to targets r.
        rim_subset = dataset
        rim_subset['PR'] = rim
        stump = stump_from_dataset(rim_subset)

    stumps.append(stump)
    print(stump)
    print(utilities.mse(dataset, stumps))

    return stumps

# Load dataset.
dataset = utilities.import_CCPP(False)
length = len(dataset.iloc[:,1])
ratio = int(length * 0.75)
train_dataset = dataset.iloc[0:ratio]
test_dataset = dataset.iloc[ratio:]

# Initialize list of classifiers and roots.
stumps = []
losses = []
m = 150

for i in range (1, m + 1):
    print("Step %d of %d\n" % (i, m))
    stumps = next_stump(train_dataset, stumps)
    losses.append(utilities.mse(train_dataset, stumps))
    # Save trees to a file.
    f = open("trees_%d.txt" % (m), "a+")
    f.write("%s:%f:%f:%f:%f\n" % (stumps[-1]['attribute'], stumps[-1]['value'], stumps[-1]['left'], stumps[-1]['right'], losses[-1]))
    f.close()
    
    # Check stopping condition(s).
    if stumps[-1]['gradient'] == math.inf:
        m = i
        break

utilities.plot_loss(m, losses, train_dataset, 'Mean Standard Error; lambda = 0.50', 'cornflowerblue')

test_losses = []
y_test = test_dataset.iloc[:,4].values.tolist()
for index, data in test_dataset.iterrows():
    test_losses.append((data['PE'] - utilities.evaluate(stumps, data)) ** 2)

utilities.plot_test(test_losses, y_test, 'Non-Library Loss for Test Dataset', 'lightseagreen')
utilities.stats(test_dataset, stumps, 'Gradient Boosting Tree Algorithm')
