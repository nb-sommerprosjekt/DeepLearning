from keras.preprocessing.text import Tokenizer, one_hot
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler


corpus_pickle = open("clean_unique_text_stemmed.pckl", "rb")
corpus = pickle.load(corpus_pickle)
(tekster, deweynr,titler) = corpus

tokenizer = Tokenizer(num_words=5000,
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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler
model_textual = Sequential([
    Dense(11222, input_shape = (5000,)),
    Activation('relu'),
    Dense(257),
    Activation('sigmoid'),
])

model_textual.compile(optimizer= 'rmsprop',
                      loss= 'binary_crossentropy',
                      metrics = ['accuracy'])

tbcallback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=True, embeddings_metadata=True)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
#LRscheduler = LearningRateScheduler()
model_textual.fit(X_train, Y_train, epochs = 10, batch_size = 64, callbacks = [tbcallback, reduceLR])
score = model_textual.evaluate(X_test, Y_test, batch_size=64)
print("\n %s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))

Y_preds = model_textual.predict(X_test)
print(Y_preds)


def precision_recall(gt,preds):
    TP=0
    FP=0
    FN=0
    for t in gt:
        if t in preds:
            TP+=1
        else:
            FN+=1
    for p in preds:
        if p not in gt:
            FP+=1
    if TP+FP==0:
        precision=0
    else:
        precision=TP/float(TP+FP)
    if TP+FN==0:
        recall=0
    else:
        recall=TP/float(TP+FN)
    return precision,recall

print ("Vår prediksjon av deweynr er -\n")
presisjon = []
recs = []
for i in range(len(Y_preds)):
    row = Y_preds[i]
    gt_deweynr =[]
    top_3=np.argsort(row)[-3:]
    predikt_deweynr = []
    for dewey in top_3:
        predikt_deweynr.append(dewey)
    (precision,recall) = precision_recall(Y_test[i], predikt_deweynr)
    presisjon.append(precision)
    recs.append(recall)
print (np.mean(np.asarray(presisjon)),np.mean(np.asarray(recs)))