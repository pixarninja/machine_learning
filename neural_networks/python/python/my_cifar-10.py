from nn import neural_network
import utils as utils

num_epochs = 1000

hidden_layers = 32
learning_rate = 0.001
cifar10_1 = neural_network('CIFAR-10', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.0001
cifar10_2 = neural_network('CIFAR-10', num_epochs, hidden_layers, learning_rate)

hidden_layers = 64
learning_rate = 0.001
cifar10_3 = neural_network('CIFAR-10', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.0001
cifar10_4 = neural_network('CIFAR-10', num_epochs, hidden_layers, learning_rate)

values = [cifar10_1, cifar10_2, cifar10_3, cifar10_4]
colors = ['olivedrab', 'mediumaquamarine', 'mediumvioletred', 'darkmagenta']
labels = ['R = 0.001, H = 32', 'R = 0.0001, H = 32', 'R = 0.001, H = 64', 'R = 0.0001, H = 64']

title = 'CIFAR-10 Loss During Training'
path = 'cifar-10/'
utils.make_dir(path)
path = path + 'loss.png'
utils.plot_together(values, colors, labels, title, path, False)
