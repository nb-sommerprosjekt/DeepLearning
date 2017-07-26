from gensim import models
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


# Leser inn pickle med label + tittel laget som list av formatet [[labels][title]]
with (open("title_label_all.pkl", "rb")) as openfile:
        title_labels=pickle.load(openfile)

# Leser inn pickle med label + tekst laget som list av formatet [[labels][tekst]]
with (open("tekst_label_all.pkl", "rb")) as openfile:
    tekst_labels = pickle.load(openfile)

#Variabler
deweynr= title_labels[0]
titler = title_labels[1]
tekst = tekst_labels[1]

#Laster inn forhåndslagd Word2Vec modell hentet fra https://github.com/Kyubyong/wordvectors
word2vec_model = models.KeyedVectors.load('pre_word2vec_no/no.bin')
print(word2vec_model['ball'].shape)


# Cleaning the texts

# Henter inn norsk stopp-ord liste fra nltk
no_stop = stopwords.words('norwegian')
print(no_stop)

corpus_word2vec = np.zeros((len(tekst),300))
print(corpus_word2vec.shape)

final_labels=[]
rows_to_delete=[]
for i in range(len(deweynr)):
    tittel_i= titler[i]
    final_labels.append(deweynr[i])
    tokens = tokenizer.tokenize(tekst[i])
    stopped_tokens = [k for k in tokens if not k in no_stop]
    count_in_vocab = 0
    s = 0
    if len(stopped_tokens) == 0:
        rows_to_delete.append(i)
        final_labels.pop(-1)
        print(tekst)
        print("sample ", i, "had no nonstops")
    else:
        for tok in stopped_tokens:
            if tok.lower() in word2vec_model.vocab:
                count_in_vocab+=1
                s+=word2vec_model[tok.lower()]
        if count_in_vocab!=0:
           corpus_word2vec[i] = s/float(count_in_vocab)
        else:
           rows_to_delete.append(i)
           final_labels.pop(-1)
           print(tekst)
           print("Sample",i,"had no word2vec")
print(len(final_labels))


from sklearn.preprocessing import MultiLabelBinarizer
X = corpus_word2vec
print(X.shape)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(final_labels)
print(Y.shape)
print(np.sum(Y, axis = 0))

textual_features = (X,Y)
mask_text = np.random.rand(len(X))<0.8
X_train = X[mask_text]
Y_train = Y[mask_text]
X_test = X[~mask_text]
Y_test = Y[~mask_text]
print(Y_test)

from keras.models import Sequential
from keras.layers import Dense, Activation
model_textual = Sequential([
    Dense(300,input_shape = (300,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model_textual.compile(optimizer= 'rmsprop',
                      loss= 'binary_crossentropy',
                      metrics = ['accuracy'])
model_textual.fit(X_train, Y_train, epochs = 1000, batch_size = 250)
score = model_textual.evaluate(X_test, Y_test, batch_size=249)
print("\n %s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))

Y_preds = model_textual.predict(X_test)



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
    print( "Predikert: ", predikt_deweynr, " Faktisk: ", Y_test[i])
print (np.mean(np.asarray(presisjon)),np.mean(np.asarray(recs)))