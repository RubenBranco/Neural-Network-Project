import tensorflow as tf
import progressbar
from utils import DataLoad, DataFormat


class RecurrentNeuralNetwork:
    def __init__(self, lstm_size, layers_n, seq_config_file):
        self.lstm_size = lstm_size
        self.num_layers = layers_n
        self.lstm_model = self.model
        self.vocab = self.vocabulary
        self.vocab_size = len(self.vocab)
        self.max_seq_size = 0
        self.input_data = None
        self.targets = None
        self.initial_state = None
        self.weights = None
        self.bias = None
        self.embedding_matrix = None
        self.embedded_inputs = None
        with open(seq_config_file) as file:
            self.max_seq_size = int(file.read())

    @property
    def model(self):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.lstm_size) for _ in range(self.num_layers)
                                            ], state_is_tuple=True)

    @property
    def vocabulary(self):
        return {'<EOS>':1, 'A':2, 'B':3, 'C':4, 'D':5, 'E':6, 'F':7, 'G':8, 'H':9, 'I':10, 'J':11, 'K':12, 'L':13, 'M':14
            ,'N':15, 'O':16, 'P':17, 'Q':18, 'R':19, 'S':20, 'T':21, 'U':22, 'V':23, 'W':24, 'X':25, 'Y':26, 'Z':27,
            'a':28, 'b':29, 'c':30, 'd':31, 'e':32, 'f':33, 'g':34, 'h':35, 'i':36, 'j':37, 'k':38, 'l':39, 'm':40,
            'n':41, 'o':42, 'p':43, 'q':44, 'r':45, 's':46, 't':47, 'u':48, 'v':49, 'w':50, 'x':51, 'y':52, 'z':53, '!':54,
            '?':55, ':':56, ',':57, ';':58, '-':59, '_':60, '^':61, '~':62, '\\':63, '|':64, '´':65, '`':66, '/':67, '*':68,
            '+':69, "'":70, '=':71, ')':72, '(':73, '&':74, '%':75, '$':76, '#':77, '"':78, '§':79, '€':80, 'º':81, 'ª':82,
            'ã':83, 'ñ':84, 'ç':85, 'á':86, 'à':87, 'é':88, 'è':89, 'ó':90, 'ò':91, 'ô':92, 'Ã':93, 'Ñ':94, 'Ç':95, 'Á':96,
            'À':97, 'É':98, 'È':99, 'Ó':100, 'Ò':101, 'Ô':102, ' ':103}

    def train(self, train_file, epochs, learning_rate, batch_size, ckpt_file = None):
        self.input_data = tf.placeholder(tf.int32, [batch_size, self.max_seq_size])
        self.targets = tf.placeholder(tf.int32, [batch_size, self.max_seq_size])
        self.initial_state = self.lstm_model.zero_state(batch_size, tf.float32)
        self.weights = tf.get_variable("weights", [self.lstm_size, self.vocab_size])
        self.embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_size, self.lstm_size])
        self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_data)
        inputs = [input_ for input_ in tf.split(self.embedded_inputs, self.max_seq_size, 1)]
        with tf.Session() as sess:


