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

    def model(self):
        hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['wl1']), self.weights['wb1']))
        hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, self.weights['wl2']), self.weights['wb2']))
        return tf.matmul(hidden_layer_2, self.weights['wout']) + self.weights['wbout']

    def train(self, epochs, learning_rate):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pmodel, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        var_init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(var_init)
            bar = progressbar.ProgressBar()
            saver = tf.train.Saver()
            if self.weights is None:
                saver.restore(session, self.weights_path)
            for _ in bar(range(epochs)):
                training_data = int(self.data.train.num_examples/100)
                for i in range(training_data):
                    image, label = self.data.train.next_batch(100)
                    _, c = session.run([optimizer, cost], feed_dict={self.x: image, self.y: label})
            correct_prediction = tf.equal(tf.argmax(self.pmodel, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print('Accuracy: ' + str(accuracy.eval({self.x: self.data.test.images, self.y: self.data.test.labels})))
            saver.save(session, self.save_path + 'model.ckpt')

    def accuracy_measure(self, session_path):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tf.train.Saver().restore(session, session_path)
            correct_prediction = tf.equal(tf.argmax(self.pmodel, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print('Accuracy: ' + str(accuracy.eval({self.x: self.data.test.images, self.y: self.data.test.labels})))


if __name__ == '__main__':
    #Example
    #nn = NeuralNetwork(256, 256, 784, 10, './model.ckpt')
    #nn.train(15, 0.001)
    #nn.accuracy_measure('./model.ckpt')



