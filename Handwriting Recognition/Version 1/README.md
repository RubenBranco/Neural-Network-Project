# nn-project

This neural network was meant as handwriting recognition, training and testing was done with MNIST dataset. The architecture, which is probably not even an actual convention in the industry, was thought of in the following way:

A base64 alphabet is done using 6 bits and that can encode any word or number. In the hidden layer, there was 6 neurons. After the dot multiplication, a matrix with 6 lines was left which was the result for each neuron. Each neuron would pass through an activation function which would yield a binary sequence of 6 bits. That could then be translated into decimal which would then identify which character it was in the base64 table. This didn't yield the best results, although it only went through about 150 epochs of training, but the efficiency starting to go down and I soon realized that I possibly should start working on something else using actual techniques which are conventions nowadays. The best accuracy result was 60%.

The way the file inputs work is: CSV file, first row being the label, or the character that is displayed in the image, the rest are the bits that comprise the image.
