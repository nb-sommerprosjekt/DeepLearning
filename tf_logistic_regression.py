'''
Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import pickle
import time
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, one_hot


tid=time.time()
corpus_pickle = open("clean_unique_text_stemmed_unique_new.pckl", "rb")
corpus = pickle.load(corpus_pickle)
(tekster, deweynr,titler) = corpus

tokenizer = Tokenizer(num_words=500,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=False,
                      split=" ",
                      char_level=False)
keras_corp = tokenizer.fit_on_texts(texts = tekster)
text_matrix = tokenizer.texts_to_matrix(tekster)

tokenizer2 = Tokenizer(num_words=257,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=False,
                      split=" ",
                      char_level=False)
keras_deweynr = tokenizer2.fit_on_texts(texts = deweynr)
dewey_matrix = tokenizer2.texts_to_matrix(deweynr)



print(text_matrix.shape)
print(type(text_matrix))
#from keras import utils.data_utils

X = text_matrix
Y = dewey_matrix
mask = np.random.rand(len(X)) < 0.8


X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


print(time.time()-tid)
#begin building and exicuting the model

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 1
display_step = 1
input_size = X.shape[1]
output_size = Y.shape[1]
num_examples=X_train.shape[0]

# tf Graph Input
x = tf.placeholder(tf.float32, [None, input_size]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, output_size]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([input_size,output_size]))
b = tf.Variable(tf.zeros([output_size]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = [X_train[i]],[Y_train[i]]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
