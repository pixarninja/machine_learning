# Modeled off of "The Sequential model API" (Keras Documentation)
# Source: https://keras.io/models/sequential/
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
import utils as utils

# Import training and testing data from TensorFlow API.
print('Loading CIFAR-10 Dataset...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalization
shape = (32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], shape[0], shape[1], shape[2]) / 255.0
x_test = x_test.reshape(x_test.shape[0], shape[0], shape[1], shape[2]) / 255.0
print('x_train shape:', x_train.shape)

# Create a Sequential base model.
model = Sequential()

# Add each layer to the model and set the shape for the nodes accordingly.
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(Conv2D(64, kernel_size=(3,3), input_shape=shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

# Compile the model and fit it with the training data.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

# Evaluate the model.
evaluation = model.evaluate(x_test, y_test)
print(evaluation)
