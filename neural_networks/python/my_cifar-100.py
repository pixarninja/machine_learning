from nn import neural_network
import utils as utils

num_epochs = 1000

hidden_layers = 32
learning_rate = 0.001
cifar100_1 = neural_network('CIFAR-100', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.0001
cifar100_2 = neural_network('CIFAR-100', num_epochs, hidden_layers, learning_rate)

hidden_layers = 64
learning_rate = 0.001
cifar100_3 = neural_network('CIFAR-100', num_epochs, hidden_layers, learning_rate)
learning_rate = 0.0001
cifar100_4 = neural_network('CIFAR-100', num_epochs, hidden_layers, learning_rate)

values = [cifar100_1, cifar100_2, cifar100_3, cifar100_4]
colors = ['salmon', 'sienna', 'cadetblue', 'darkslateblue']
labels = ['R = 0.001, H = 32', 'R = 0.0001, H = 32', 'R = 0.001, H = 64', 'R = 0.0001, H = 64']

title = 'CIFAR-100 Loss During Training'
path = 'cifar-100/'
utils.make_dir(path)
path = path + 'loss.png'
utils.plot_together(values, colors, labels, title, path, False)
