import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from gensim import models
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


#######################################################################
#------------------Pre-processing -----------------------------------#
#--------------------------------------------------------------------#

# Leser inn pickle med label + tittel laget som list av formatet [[labels][title]]
with (open("title_label_all.pckl", "rb")) as openfile:
        title_labels=pickle.load(openfile)

# Leser inn pickle med label + tekst laget som list av formatet [[labels][tekst]]
with (open("tekst_label_all.pckl", "rb")) as openfile:
    tekst_labels = pickle.load(openfile)

#Variabler
deweynr= title_labels[0]
titler = title_labels[1]
tekst = tekst_labels[1]

#Laster inn forhåndslagd Word2Vec modell hentet fra https://github.com/Kyubyong/wordvectors
word2vec_model = models.KeyedVectors.load('pre_word2vec_no/no.bin')
#print(word2vec_model['ball'].shape)


# Cleaning the texts

# Henter inn norsk stopp-ord liste fra nltk
no_stop = stopwords.words('norwegian')
#print(no_stop)

corpus_word2vec = np.zeros((len(tekst),300))
#print(corpus_word2vec.shape)

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
#print(len(final_labels))




X = corpus_word2vec
#print(X.shape)
lb = LabelBinarizer()
Y = lb.fit_transform(final_labels)
#print('-------------')
#print(Y.shape)
#print('-------------')
#print(np.sum(Y, axis = 0))
textual_features = (X,Y)

feature_pickle = open('textual_features.pckl','wb')
pickle.dump(textual_features, feature_pickle)
feature_pickle.close()