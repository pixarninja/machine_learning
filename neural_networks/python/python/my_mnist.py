from nn import neural_network
import utils as utils

num_epochs = 1000

hidden_layers = 16
learning_rate = 0.1
mnist_1 = neural_network('MNIST', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.01
mnist_2 = neural_network('MNIST', num_epochs, hidden_layers, learning_rate)

hidden_layers = 32
learning_rate = 0.1
mnist_3 = neural_network('MNIST', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.01
mnist_4 = neural_network('MNIST', num_epochs, hidden_layers, learning_rate)

values = [mnist_1, mnist_2, mnist_3, mnist_4]
colors = ['cornflowerblue', 'mediumseagreen', 'indianred', 'darkgoldenrod']
labels = ['R = 0.1, H = 16', 'R = 0.01, H = 16', 'R = 0.1, H = 32', 'R = 0.01, H = 32']

title = 'MNIST Loss During Training'
path = 'mnist/'
utils.make_dir(path)
path = path + 'loss.png'
utils.plot_together(values, colors, labels, title, path, False)
