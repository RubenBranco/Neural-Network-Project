import numpy as np
import math
import base64

def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))

def init_weights(input_size,neuron_size):
    initial_weights = []
    r = 4*math.sqrt(6/(input_size+1))
    for _ in range(neuron_size):
        weights = []
        for i in range(input_size):
            weights.append(np.random.uniform(-r, r))
        initial_weights.append(weights)
    return np.mat(initial_weights)


def train_nn(train_file, input_size, neuron_size):
    weights = init_weights(input_size, neuron_size)
    with open('../mnist_train.csv') as training_file:
        for line in training_file:

