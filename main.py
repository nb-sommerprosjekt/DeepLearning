import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from preprocessing import final_labels


#-----------------------------------------------------------------------#
#--------------------Trening - Test-------------------------------------#
#----------------------------------------------------------------------#

feature_pickle = open('textual_features.pckl','rb')
textual_features = pickle.load(feature_pickle)
feature_pickle.close()

(X, Y) = textual_features
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state= 200, stratify = Y)
mask_text = np.random.rand(len(X))<0.8


#--------------------Building Keras models ----------#
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard, ReduceLROnPlateau
model_textual = Sequential([
    Dense(300,input_shape = (300,)),
    Activation('relu'),
    Dense(304),
    Activation('softmax'),
])

model_textual.compile(optimizer= 'rmsprop',
                      loss= 'binary_crossentropy',
                      metrics = ['accuracy'])

tbcallback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1000, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=True, embeddings_metadata=True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
model_textual.fit(X_train, Y_train, epochs = 1000, batch_size = 1000, callbacks = [tbcallback, reduceLR])
score = model_textual.evaluate(X_test, Y_test, batch_size=1000)
print("\n %s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))


#-------------------Testing model-------------------------#
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
lb = LabelBinarizer()
Y_test_ikkebinary = lb.fit_transform(final_labels)
Y_test_ikkebinary = lb.inverse_transform(Y_test)
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
    print( "Predikert: ", predikt_deweynr, " Faktisk: ", Y_test_ikkebinary[i])
print (np.mean(np.asarray(presisjon)),np.mean(np.asarray(recs)))

# #print(mlb.inverse_transform(Y_test))