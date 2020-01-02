# Machine Learning Resources

This is a small repository of basic machine learning resources.
The recommended workflow for these resources is as follows:
Regression, Decision Trees, Logisitic Regression,
and finally Neural Networks. Each repository provides a LaTeX
report and compile script that discusses the results of what was
developed.

### Regression

Regression is a tool for solving an optimization problem,
specifically when the result is numerical.
Different types of regression solve the problem in different ways.
Linear regression attempts to separate the data linearly over the
feature space. The result is a hyperplane that can be used to
make a prediction for unseen input.
Here we look at Least Squares Linear Regression, and Ridge, Lasso,
and Elastic Net Regression methods.
Topics discussed in the report include:

 * Objective Function
 * Stochastic Gradient Descent
 * Least Squares Linear Regression (LSLR)
 * Ridge, Lasso, and Elastic Net Regression
 * Python Implementation

### Decision Trees
 Although a tree is easy and fast to build, the
predictive power of an individual tree is usually very low.
The tree is likely to overfit and not
produce good results on unseen data.
Researchers found that using a collection of trees can produce
better results than a single tree alone.
In general, there are two ways to develop such a collection:
Bagging and Boosting.
Here we look at Boosting with the simplest form of trees, stumps.
Topics discussed in the report include:
 * Stumps
 * Splitting
 * Boosting Regression Trees
 * Python and R Implementations

### Logistic Regression
Logistic regression in general is a binary classifier,
although it can be extended to multiple classes
(called multinomial logistic regression).
Logistic regression is more powerful than linear regression
methods, because it's activation function is dynamic and tunable.
Here we look at a simple Logistic Regression approach.
Topics discussed in the report include:
 * Log-Odds
 * Maximum Likelihood Approach
 * Gradient Descent and Netwon-Raphson Techniques
 * Python and R Implementations

### Neural Networks
A basic artificial neural network (ANN) has three layers:
input, hidden and output.
The network trains a set of weights based on the difference
between expected and obtained outputs, using backpropagation.
At the end of training, the network should be able to
make a prediction with some accuracy given new input data.
Although this design has been proven to be a Universal Classifier,
there are many extentions and optimizations
(such as Convolutional Neural Networks) that make processing
for certain problems easier and faster.
Here we look at making a simple ANN from scratch, and compare our
results to CNN's with varying numbers of layers.
Topics discussed in the report include:
 * Backpropogation
 * Artificial Neural Network (ANN)
 * Convolutional Nueral Network (CNN)
 * MNIST and CIFAR Datasets
 * Python and R Implementations
