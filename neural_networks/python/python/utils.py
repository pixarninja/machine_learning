from collections import defaultdict
import copy as copy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os as os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds

# Make a directory if it doesn't exist.
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Encode data into categories.
def encode(data):
    return np.array(to_categorical(data))

# Decode data from categories.
def decode(data):
    return np.argmax(data)

# Shuffle two arrays in the same way.
def shuffle_data(x, y):
    for i in reversed(range(1, len(x))):
        # pick an element in x[:i] with which to exchange x[i]
        j = np.random.randint(0, i)
        x_i = np.array(x[i])
        x_j = np.array(x[j])
        x[i], x[j] = x_j, x_i
        y_i = np.array(y[i])
        y_j = np.array(y[j])
        y[i], y[j] = y_j, y_i

    return x, y

# Save test image to check import.
def test_import(data, path):
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(data)
    fig.savefig(path)
    plt.clf()
    print('Saved image to: ' + path)

# Import MNIST training and testing data from TensorFlow API.
def import_MNIST():
    print('Loading MNIST Dataset...')
    (x_train_d, y_train), (x_test_d, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = []
    shape = x_train_d.shape
    for i in range(shape[0]):
        x_train.append(x_train_d[i].reshape(shape[1] * shape[2]) / 255.0)
    x_train, y_train = shuffle_data(x_train, y_train)

    x_test = []
    shape = x_test_d.shape
    for i in range(shape[0]):
        x_test.append(x_test_d[i].reshape(shape[1] * shape[2]) / 255.0)
    x_test, y_test = shuffle_data(x_test, y_test)

    # Save test import.
    path = 'mnist/'
    make_dir(path)
    test_import(x_train[0].reshape(28, 28), path + str(y_train[0]) + '.png')

    return np.array(x_train).T, encode(y_train).T, np.array(x_test).T, encode(y_test).T

# Import CIFAR training and testing data from stored files.
def import_CIFAR(cifar_type):
    if cifar_type == 10:
        path = 'cifar-10'
    else:
        path = 'cifar-100'

    x_train, y_train, x_test, y_test, meta = helper_CIFAR(path)
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test = shuffle_data(x_test, y_test)

    return np.array(x_train).T, encode(y_train).T, np.array(x_test).T, encode(y_test).T

# Helper function for splitting CIFAR training and testing data.
def helper_CIFAR(cifar_type):
    prefix = '../../../source/'
    images = None
    labels = None

    if cifar_type == 'cifar-10':
        meta_postfix = '/batches.meta'
        with open('{0}{1}{2}'.format(prefix, cifar_type, meta_postfix), 'rb') as f:
            meta = pickle.load(f, encoding='bytes')

        data_postfix = '/data_batch_'
        for i in range(1, 6):
            path = '{0}{1}{2}{3}'.format(prefix, cifar_type, data_postfix, i)
            print('Loading path: ' + path)
            with open(path, 'rb') as f:
                pickle_dict = pickle.load(f, encoding='bytes')

            pickle_images = np.array(pickle_dict[b'data'], dtype=np.uint8)
            if images is not None:
                images = np.concatenate([images, pickle_images])
            else:
                images = pickle_images

            pickle_labels = np.array(pickle_dict[b'labels'], dtype=np.uint8)
            if labels is not None:
                labels = np.concatenate([labels, pickle_labels])
            else:
                labels = pickle_labels
        
        data_postfix = '/test_batch'
        path = '{0}{1}{2}'.format(prefix, cifar_type, data_postfix)
        print('Loading path: ' + path)
        with open(path, 'rb') as f:
            pickle_dict = pickle.load(f, encoding='bytes')

        pickle_images = np.array(pickle_dict[b'data'], dtype=np.uint8)
        if images is not None:
            images = np.concatenate([images, pickle_images])
        else:
            images = pickle_images

        pickle_labels = np.array(pickle_dict[b'labels'], dtype=np.uint8)
        if labels is not None:
            labels = np.concatenate([labels, pickle_labels])
        else:
            labels = pickle_labels
    else:
        meta_postfix = '/meta'
        with open('{0}{1}{2}'.format(prefix, cifar_type, meta_postfix), 'rb') as f:
            meta = pickle.load(f, encoding='bytes')

        data_postfix = '/train'
        path = '{0}{1}{2}'.format(prefix, cifar_type, data_postfix)
        print('Loading path: ' + path)
        with open(path, 'rb') as f:
            pickle_dict = pickle.load(f, encoding='bytes')

        for key, value in pickle_dict.items():
            print(key)

        pickle_images = np.array(pickle_dict[b'data'], dtype=np.uint8)
        if images is not None:
            images = np.concatenate([images, pickle_images])
        else:
            images = pickle_images

        pickle_labels = np.array(pickle_dict[b'fine_labels'], dtype=np.uint8)
        if labels is not None:
            labels = np.concatenate([labels, pickle_labels])
        else:
            labels = pickle_labels

        data_postfix = '/test'
        path = '{0}{1}{2}'.format(prefix, cifar_type, data_postfix)
        print('Loading path: ' + path)
        with open(path, 'rb') as f:
            pickle_dict = pickle.load(f, encoding='bytes')

        pickle_images = np.array(pickle_dict[b'data'], dtype=np.uint8)
        if images is not None:
            images = np.concatenate([images, pickle_images])
        else:
            images = pickle_images

        pickle_labels = np.array(pickle_dict[b'fine_labels'], dtype=np.uint8)
        if labels is not None:
            labels = np.concatenate([labels, pickle_labels])
        else:
            labels = pickle_labels

    # Save test import.
    path = cifar_type + '/'
    make_dir(path)
    index = np.random.randint(0, len(images))
    test_import(images[index].reshape(3, 32, 32).transpose(1,2,0), path + str(labels[index]) + '.png')

    # Reshape image data.
    pixel_data = []
    for i in range(len(images)):
        pixels = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        transformed = pixels.sum(axis=2)
        pixel_data.append(transformed.reshape(32 * 32))

    print(np.array(pixel_data).shape)

    # Split into training and testing data.
    split = 50000
    x_train, x_test = np.array(pixel_data)[:split], np.array(pixel_data)[split:]
    y_train, y_test = labels[:split], labels[split:]

    print('Metadata: ', meta)
    print('Images: ', images.shape)
    print('Labels: ', labels.shape)
    return x_train / (3 * 255.0), y_train, x_test / (3 * 255.0), y_test, meta

# Definition for plotting values.
def plot_together(values, colors, labels, title, path, fit_flag):
    samples = len(values[0])
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    patches = []
    
    # Plot values and fits.
    for i in range(len(values)):
        avg = np.average(values[i])
        plt.plot(x_axis, values[i], color=colors[i], alpha=0.5)
        if fit_flag:
            plt.plot(np.unique(x), np.poly1d(np.polyfit(x, values[i], 1))(np.unique(x)), color=colors[i], linestyle='--')
        patches.append(mpatches.Patch(color=colors[i]))
        print(title + '[' + str(i) + ']: ' + str(avg))
    
    # Finish plot.
    plt.legend(patches, labels, loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
