from sklearn.feature_extraction.text import CountVectorizer
from gensim import models
import pickle
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.snowball import NorwegianStemmer

# Leser inn pickle med label + tittel laget som list av formatet [[labels][title]]
#title_labels = []
with (open("title_label_all.pkl", "rb")) as openfile:
        title_labels=pickle.load(openfile)

# Leser inn pickle med label + tekst laget som list av formatet [[labels][tekst]]
with (open("tekst_label_all.pkl", "rb")) as openfile:
    tekst_labels = pickle.load(openfile)

#Variabler
deweynr= title_labels[0]
titler = title_labels[1]
tekst = tekst_labels[1]

# Laster inn forhåndslagd Word2Vec modell hentet fra https://github.com/Kyubyong/wordvectors
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
#print(Y)

#for i in range(0, len(deweynr)):
#     #print (value)
#     value =re.sub('[^ÆØÅæøåa-zA-z]',' ',value)
#     #print(value)
#     value = value.lower()
#     #print(value)
#     value = value.split()
#     #print(value)
#     NorStem = NorwegianStemmer()
#     value = [NorStem.stem(word) for word in value if not word in set(stopwords.words('norwegian'))]
#     #print(value)
#     value = ' '.join(value)
#     #print(value)
#     corpus.append(value)
#     deweynr.append(key)
#
# #print(deweynr)
# print ((deweynr).shape)
# print(len(corpus).shape)
#
# # Creating the Bag of Words model
