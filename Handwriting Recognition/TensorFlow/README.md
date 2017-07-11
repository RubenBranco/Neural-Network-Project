# nn-project

Redone NN using TensorFlow, this time accuracy jumped to the 98% ball park. Examples on how to run it are on the last section of the file. 

It has two hidden layers but can be rewritten to have more, of course. The __init__ function takes 6 parameters, one of them optional. First two are the hidden layers size(number), followed by the size of the input (in this case it's using MNIST dataset so 28 * 28 = 784, but if others is used can also be put in, just have to change the self.data), and then the saving path for the session and if there's no session already done then weights are randomly initialized using Xavier's initializer.

nn.train takes 2 arguments, epochs and learning_rate, and will use progressbar to give a nice ETA on how long the training will take. In the end the accuracy is displayed.

Can also test the accuracy outside of training using the accuracy_measure function.
