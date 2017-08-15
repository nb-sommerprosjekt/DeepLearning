import tensorflow as tf
import numpy as np
import pickle
import time
from keras.preprocessing.text import Tokenizer, one_hot,text_to_word_sequence
from sklearn.model_selection import train_test_split
tf.logging.set_verbosity(tf.logging.INFO)

#PARAMETERS

TEXT_LENGTH = 2000



def reshape_list_to_matrix(sequence):
    # sequence = np.array(sequence)
    # sequence.resize(10000)
    # sequence.tolist()
    # for i in range(len(sequence)):
    #     temp=str(sequence[i]).strip()
    #     sequence[i]=int(temp) if temp else 0
    if len(sequence)>TEXT_LENGTH:
        sequence=sequence[:TEXT_LENGTH]
    else:
        while len(sequence)<TEXT_LENGTH:
            sequence.append([0])

    return sequence



#Load in the corpus
tid=time.time()
corpus_pickle = open("clean_unique_text_stemmed_unique_new.pckl", "rb")
corpus = pickle.load(corpus_pickle)
(tekster, deweynr,titler) = corpus

test=corpus[0][1]

seq=text_to_word_sequence(test)




tokenizer2 = Tokenizer(num_words=257,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=False,
                      split=" ",
                      char_level=False)
keras_deweynr = tokenizer2.fit_on_texts(texts = deweynr)
dewey_matrix = tokenizer2.texts_to_matrix(deweynr)


# prøv med den andre tokenizeren.
tokenizer = Tokenizer(num_words=3000,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=False,
                      split=" ",
                      char_level=False)
keras_corp = tokenizer.fit_on_texts(texts = tekster)
#text_matrix = tokenizer.texts_to_matrix(tekster)



#from keras import utils.data_utils

X = tekster
Y = dewey_matrix
mask = np.random.rand(len(X)) < 0.8

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=1)

# for i, stempequence in enumerate(tokenizer.texts_to_sequences_generator(X_train)):
#     temp = reshape_list_to_matrix(tokenizer, tokenizer.sequences_to_matrix(sequence))
#     train_step.run(feed_dict={x: [temp], y_: [Y_train[i]], keep_prob: 0.5})

        # temp=tokenizer.texts_to_sequences(test)
        # print(len(temp))
        # temp=reshape_list_to_matrix(temp)
        # temp=tokenizer.sequences_to_matrix(temp)
        # temp=np.asarray(temp)
        # print(temp.shape)
print(time.time()-tid)

#Build the actual network

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x,(-1,TEXT_LENGTH,3000,1))

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 4])
  b_conv1 = bias_variable([4])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 4,8])
  b_conv2 = bias_variable([8])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([500*750*8, 1000])#2500*750*64
  b_fc1 = bias_variable([1000])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 500*750*8])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1000, 257])
  b_fc2 = bias_variable([257])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



  # Import data
nr_words = 3000
labels = Y_train.shape[1]
epochs=1
display_step=0

x = tf.placeholder(tf.float32, [None, TEXT_LENGTH, 3000])


# Define loss and optimizer
y_ = (tf.placeholder(tf.float32, [None,labels]))

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

cost = tf.reduce_mean(-tf.reduce_sum(y_conv*tf.log(y_), reduction_indices=1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0.
        print("epoch nr {}".format(epoch))
        for i in range(len(X_train)):

            sequence=tokenizer.texts_to_sequences(X_train[i])
            sequence = reshape_list_to_matrix(sequence)
            sequence = tokenizer.sequences_to_matrix(sequence)
            print(sequence.shape)


            train_step.run(feed_dict={x: [sequence], y_: [Y_train[i]], keep_prob: 0.5})
        #     _, c = sess.run([train_step, cost], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        #     avg_cost += c/num_examples
        # if (epoch+1) % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        acc=0
        for i,sequence in enumerate(tokenizer.texts_to_sequences_generator(X_test)):
            acc+=accuracy.eval(feed_dict={x: [reshape_list_to_matrix(tokenizer,sequence)], y_: [Y_train[i]], keep_prob: 1.0})

        print(epoch,' accuracy %g' % acc/len(X_test))






