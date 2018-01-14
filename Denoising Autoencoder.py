from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Setting up neural network...")

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.04   # 0.04
num_epochs = 25
batch_size = 200
num_display_iterations = 2
# the first test is executed after the first epoch, regardless of num_tests
num_tests = 5

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])), 
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Corruption function
def corrupt(x):
    # generate numpy array with gaussian noise (random normal distribution) with stddev = 0.2 and the shape size = 784
    # stddev = 0.4 for heavy noise (min 50 epochs) and stddev = 0.2 for normal noise
    noise = np.random.normal(0, 0.2, 784)
    return x + noise

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with tanh activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with tanh activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with tanh activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with tanh activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
X = tf.placeholder("float", [None, num_input])  # tf Graph input (only pictures)
Z = encoder(corrupt(X))    # Latent representation. Input of Z is a corrupted version of X
Y = decoder(Z)             # Output                     

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(X - Y, 2))    #  x - x'
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

def testAE():
    print("\nTesting Autoencoder...")

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 10
    canvas_orig = np.empty((28, 28 * n))
    canvas_corrupt = np.empty((28, 28 * n))
    canvas_recon = np.empty((28, 28 * n))
    
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    
    # Encode and decode the digit image and determine the loss
    g, l = sess.run([Y, loss], feed_dict={X: batch_x})

    for i in range(n):
        # Original images
        canvas_orig[0: 28, i * 28: (i + 1) * 28] = batch_x[i].reshape([28, 28])
        
        # Crrupted images
        canvas_corrupt[0: 28, i * 28: (i + 1) * 28] = corrupt(batch_x[i]).reshape([28, 28]) 
        
        # Reconstructed images
        canvas_recon[0: 28, i * 28: (i + 1) * 28] = g[i].reshape([28, 28])    
    
    print("Original Images")     
    plt.figure(figsize=(n, 1))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()
    
    print("Corrupted Images")     
    plt.figure(figsize=(n, 1))
    plt.imshow(canvas_corrupt, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, 1))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
    
    print("Test loss: " + str(l) + "\n")

print("\nTraining the Autoencoder...")

num_iterations = mnist.train.num_examples // batch_size   # Number of iterations in one epoch
absolute_iterations_count = 1    

# Training
for epoch in range(1, num_epochs + 1):
    print("Epoch " + str(epoch) + ":")

    for i in range(1, num_iterations + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        
        # Display logs per step
        if i % (num_iterations // num_display_iterations) == 0:
            print("\tIteration " + str(absolute_iterations_count) + "  \tLoss: " + str(l))

        absolute_iterations_count += 1
            
    if epoch % (num_epochs // num_tests) == 0 or epoch == 1:
        testAE()

print("Training completed")