import tensorflow as tf

import numpy as np
import pickle
import time
from keras.preprocessing.text import Tokenizer, one_hot

#Load in the corpus
tid=time.time()
corpus_pickle = open("clean_unique_text_stemmed_unique_new.pckl", "rb")
corpus = pickle.load(corpus_pickle)
(tekster, deweynr,titler) = corpus

tokenizer = Tokenizer(num_words=4000,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)
keras_corp = tokenizer.fit_on_texts(texts = tekster)
text_matrix = tokenizer.texts_to_matrix(tekster)

tokenizer2 = Tokenizer(num_words=257,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
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


#Build the actual network


nr_words=X_train.shape[1]
labels=Y_train.shape[1]


x = tf.placeholder(tf.float32,[None,nr_words])

W = tf.Variable(tf.zeros([nr_words,labels]))
b = tf.Variable(tf.zeros([labels]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = (tf.placeholder(tf.float32, [None,labels]))


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step =tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(X_train.shape[0]):
    batch_xs, batch_ys = [X_train[i]], [Y_train[i]]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}))