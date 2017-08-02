import tensorflow as tf
import progressbar
from utils import DataLoad, DataFormat
import numpy as np


class RecurrentNeuralNetwork:
    def __init__(self, lstm_size, layers_n, seq_config_file, batch_size):
        self.lstm_size = lstm_size
        self.num_layers = layers_n
        self.lstm_model = self.model
        self.vocab = self.vocabulary
        self.vocab_size = len(self.vocab)
        self.max_seq_size = DataFormat.seq_len(seq_config_file)
        self.input_data = tf.placeholder(tf.int32, [batch_size, self.max_seq_size])
        self.targets = None
        self.initial_state = self.lstm_model.zero_state(batch_size, tf.float32)
        self.weights = None
        self.bias = None
        self.embedding_matrix = None
        self.embedded_inputs = None
        self.logits = None
        self.probs = None
        self.cost = None
        self.train_op = None
        self.final_state = None
        self.lr = None

    @property
    def model(self):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.lstm_size) for _ in range(self.num_layers)
                                            ], state_is_tuple=True)

    @property
    def vocabulary(self):
        return 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.:,;-_^~><\\|´`/*+\'«»=÷)({}[]ößü&%$#"€£@§¡¿ºªãñçâáàâéèêóòôõúùíìîýỳÃÑÇÂÂÁÀÉÈÊÓÒÔÍÌÚÙÝỲÕÎ 0123456789'

    def train(self, train_file, epochs, learning_rate, batch_size, grad_clip, ckpt_file=None):
        self.targets = tf.placeholder(tf.int32, [batch_size, self.max_seq_size])
        self.weights = tf.get_variable("weights", [self.lstm_size, self.vocab_size])
        self.embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_size, self.lstm_size])
        self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_data)
        masked_embedded = tf.concat([tf.zeros([1, 1]), tf.ones([self.embedding_matrix.get_shape()[0]-1, 1])], 0)
        self.embedded_inputs = tf.matmul(self.embedded_inputs, tf.nn.embedding_lookup(masked_embedded, self.input_data))
        inputs = [input_ for input_ in tf.split(self.embedded_inputs, self.max_seq_size, 1)]
        print(inputs)

        def loop(self_obj, prev, _):
            prev = tf.matmul(prev, self_obj.weights) + self_obj.bias
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(self_obj.embedding_matrix, prev_symbol)

        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.lstm_model,
                                                                    loop_function=loop)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.lstm_size])

        self.logits = tf.matmul(output, self.weights) + self.bias
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * self.max_seq_size])])
        self.cost = tf.reduce_sum(loss) / batch_size / self.max_seq_size
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / batch_size / self.max_seq_size
        self.final_state = last_state
        self.lr = learning_rate
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            bar = progressbar.ProgressBar()
            if ckpt_file is not None:
                saver.restore(sess, ckpt_file)
            for _ in bar(range(epochs)):
                data = DataLoad(50, train_file, 1000000)
                batch = data.initial_batch()
                state = self.initial_state
                while batch is not None:
                    for sentence in batch:
                        data_format = DataFormat()
                        x,y = data_format.one_hot(sentence), data_format.one_hot(sentence)
                        feed_dict = {self.input_data:x, self.targets:y}
                        for i, (c, h) in enumerate(self.initial_state):
                            feed_dict[c] = state[i].c
                            feed_dict[h] = state[i].h
                        train_loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], feed_dict)

    def sample(self, session, n_chars, sample_type, prime_text=' '):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, session)
            state = sess.run(self.lstm_model.zero_state(1, tf.float32))
            data_format = DataFormat()
            for char in prime_text[:-1]:
                feed_dict = {self.input_data: data_format.one_hot(char), self.initial_state: state}
                [state] = sess.run([self.final_state], feed_dict)

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return np.searchsorted(t, np.random.rand(1)*s)

            text = prime_text
            prev_char = prime_text[-1]
            for _ in range(n_chars):
                feed_dict = {self.input_data: data_format.one_hot(prev_char), self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed_dict)
                p = probs[0]
                if sample_type == 0:
                    sample = np.argmax(p)
                elif sample_type == 2:
                    if prev_char == ' ':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else:
                    sample = weighted_pick(p)

                pred = self.vocab[int(sample)]
                text += pred
                prev_char = pred
            return text.encode('utf-8')

if __name__ == '__main__':
    nn = RecurrentNeuralNetwork(1005, 2, 'maxseqsize.config', 50)
    nn.train('/home/ruben/PycharmProjects/Ruben/twitch.log', 2, 0.001, 50, 5.0)