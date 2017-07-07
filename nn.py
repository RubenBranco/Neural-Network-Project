import numpy as np
import math
base = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'


def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))


def init_weights(input_size, neuron_size):
    initial_weights = []
    r = 4*math.sqrt(6/(input_size+1))
    for _ in range(neuron_size):
        weights = []
        for i in range(input_size+1):
            weights.append(np.random.uniform(-r, r))
        initial_weights.append(weights)
    return np.mat(initial_weights)


def perceptron(weights_matrix, input_matrix):
    hidden_inputs = weights_matrix * input_matrix
    binary_num = ''
    for i in range(len(hidden_inputs)):
        activation_function_output = round(sigmoid(hidden_inputs.item(i)))
        binary_num += str(activation_function_output)
    value = int(binary_num, 2)
    return base[value]


def train_nn(train_file, input_size, neuron_size, learning_rate, path, weights_file=False):
    weights = []
    if weights_file:
        weights = np.load(path+'weights.npy')
    else:
        weights = init_weights(input_size, neuron_size)
    with open(train_file) as training_file:
        for line in training_file:
            input_matrix = []
            split_line = line.split(',')
            label = split_line[0]
            for i in range(1,len(split_line)):
                input_matrix.append([int(split_line[i])])
            input_matrix.append([1])
            percepton_guess = perceptron(weights, np.mat(input_matrix))
            percepton_guess_binary = bin(base.index(percepton_guess))[2:]
            label_binary = bin(base.index(label))[2:]
            percepton_guess_binary = '0'*(6-len(percepton_guess_binary)) + percepton_guess_binary
            label_binary = '0'*(6-len(label_binary)) + label_binary
            for i in range(neuron_size):
                error = int(label_binary[i]) - int(percepton_guess_binary[i])
                for j in range(input_size + 1):
                    weights[i, j] += learning_rate * error * input_matrix[j][0]
            np.save(path+'weights.npy', weights)


def train_loop(epoch):
    for _ in range(epoch):
        train_nn('/Users/rubenbranco/Desktop/neuralnetwork/mnist_train.csv', 784, 6, 0.1,'/Users/rubenbranco/Desktop/neuralnetwork/','/Users/rubenbranco/Desktop/neuralnetwork/weights.npy')

def guess():
    line = ''
    with open('/Users/rubenbranco/Desktop/neuralnetwork/mnist_test.csv') as file:
        for _ in range(8):
            line = file.readline()
    split_line = line.split(',')
    num = split_line[0]
    weights = np.load('/Users/rubenbranco/Desktop/neuralnetwork/weights.npy')
    input_matrix = []
    for i in range(1, len(split_line)):
        input_matrix.append([int(split_line[i])])
    input_matrix.append([1])
    res = perceptron(weights, np.mat(input_matrix))
    return num, res