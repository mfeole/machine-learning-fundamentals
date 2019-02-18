import numpy as np
import h5py
import time
import copy
from random import randint
import os

# Load MNIST dataset
if os.path.exists(os.path.join(os.getcwd(), 'data\\MNISTdata.hdf5')):
    MNIST_data = h5py.File(os.path.join(os.getcwd(), 'data\\MNISTdata.hdf5'), 'r')
elif os.path.exists(os.path.join(os.getcwd(), '..\\data\\MNISTdata.hdf5')):
    MNIST_data = h5py.File(os.path.join(os.getcwd(), '..\\data\\MNISTdata.hdf5'), 'r')
else:
    print('\nError: Run script from main path or script folder path.\n')
    raise NotADirectoryError

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array( MNIST_data['y_test'][:, 0]))

MNIST_data.close()

# image size
num_inputs = 28*28
# number of digits
num_outputs = 10

# Number of units in the hidden layer
num_hidden_units = 35

# Model parameters, random initialization
model = {}
model['W'] = np.random.randn(num_hidden_units, num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(num_hidden_units) / np.sqrt(num_hidden_units)
model['C'] = np.random.randn(num_outputs, num_hidden_units) / np.sqrt(num_hidden_units)
model['b2'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)


# Function definitions: softmax, ReLU and the derivative of ReLU

def softmax_function(z):
    ZZ = np.exp(z) / np.sum(np.exp(z))
    return ZZ


def ReLU(z):
    # Implementation of ReLU function
    relu = copy.deepcopy(z)
    relu[z <= 0.0] = 0.0
    return relu


def deriv_ReLU(z):
    # Hand derivative of the ReLU function
    # put a value of 0.5 when z == 0
    deriv = copy.deepcopy(z)
    deriv[z > 0.0] = 1.0
    deriv[z < 0.0] = 0.0
    deriv[z == 0.0] = 0.5
    return deriv


# Algorithms for Forward and Backward steps

def forward(x, model):
    # Forward step with one hidden layer
    Z = np.dot(model['W'], x) + model['b1']
    H = ReLU(Z)
    U = np.dot(model['C'], H) + model['b2']
    p = softmax_function(U)
    return p, Z, H


def backward(x, y, p, Z, H, model, model_grads):
    # Backward step
    drho_dU = p
    drho_dU[y] -= 1.0
    drho_db2 = drho_dU
    drho_dC = np.dot(drho_dU.reshape((len(drho_dU), 1)),
                     np.transpose(H.reshape((len(H), 1))))
    delta = np.dot(np.transpose(model['C']), drho_dU)

    drho_db1 = np.multiply(delta, deriv_ReLU(Z))
    drho_dW = np.dot(drho_db1.reshape((len(drho_db1), 1)),
                     np.transpose(x.reshape((len(x), 1))))

    model_grads['W'] = drho_dW
    model_grads['b1'] = drho_db1
    model_grads['C'] = drho_dC
    model_grads['b2'] = drho_db2

    return model_grads


# Model training

time1 = time.time()
LR = 0.01
num_epochs = 20
for epochs in range(num_epochs):
    print("Epoch {}/{}... ".format(epochs + 1, num_epochs), end='')
    # Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001

    total_correct = 0

    for n in range(len(x_train)):
        n_random = randint(0, len(x_train) - 1)
        y = y_train[n_random]
        x = x_train[n_random][:]

        p, Z, H = forward(x, model)

        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1

        model_grads = backward(x, y, p, Z, H, model, model_grads)

        # Updating parameters
        model['W'] = model['W'] - LR * model_grads['W']
        model['b1'] = model['b1'] - LR * model_grads['b1']
        model['C'] = model['C'] - LR * model_grads['C']
        model['b2'] = model['b2'] - LR * model_grads['b2']

    partial_acc = total_correct / np.float(len(x_train))
    print("train accuracy: {:1.4f}".format(partial_acc))

time2 = time.time()
print("\nTraining time: {:3.2f} seconds.\n".format(time2 - time1))


# Computing accuracy on test set

total_correct = 0

for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]

    p, Z, H = forward(x, model)

    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1

accuracy = total_correct / np.float(len(x_test))
print("Accuracy on test set: {:1.4f}\n".format(accuracy))
