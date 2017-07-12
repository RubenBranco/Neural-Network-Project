import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import progressbar


class NeuralNetwork:
    def __init__(self, n_layer1, n_layer2, n_input, n_classes, save_path, weights_path=None):
        self.hidden1 = n_layer1
        self.hidden2 = n_layer2
        self.n_input = n_input
        self.n_classes = n_classes
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])
        self.save_path = save_path
        if weights_path is None:
            self.weights = {
                'wl1': tf.get_variable("wl1", shape = [n_input, n_layer1], initializer = tf.contrib.layers.xavier_initializer()),
                'wl2': tf.get_variable("wl2", shape = [n_layer1, n_layer2], initializer = tf.contrib.layers.xavier_initializer()),
                'wout': tf.get_variable("wout", shape = [n_layer2, n_classes], initializer = tf.contrib.layers.xavier_initializer()),
                'wb1': tf.get_variable("wb1", shape = [n_layer1], initializer = tf.contrib.layers.xavier_initializer()),
                'wb2': tf.get_variable("wb2", shape = [n_layer2], initializer = tf.contrib.layers.xavier_initializer()),
                'wbout': tf.get_variable('wbout', shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())
            }
        else:
            
            self.weights = None
            self.weights_path = weights_path
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.pmodel = self.model()



if __name__ == '__main__':
    #Example
    #nn = NeuralNetwork(256, 256, 784, 10, './model.ckpt')
    #nn.train(15, 0.001)
    #nn.accuracy_measure('./model.ckpt')



