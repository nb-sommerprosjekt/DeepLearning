import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from gensim import models
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import time
start = time.time()
from nltk.stem import snowball
#######################################################################
#------------------Pre-processing -----------------------------------#
#--------------------------------------------------------------------#

# Leser inn pickle med label + tittel laget som list av formatet [[labels][title]]
with (open("title_label_knut0708.pckl", "rb")) as openfile:
        title_labels=pickle.load(openfile)

# Leser inn pickle med label + tekst laget som list av formatet [[labels][tekst]]
with (open("tekst_label_all_knut0708.pckl", "rb")) as openfile:
    tekst_labels = pickle.load(openfile)


#Variabler
deweynr= title_labels[0]
titler = title_labels[1]
tekst = tekst_labels[1]


# Vasking av tekst, gjennom følgende prosess: lowercase, stopp-ord fjerning, stemming og deretter pickles filen.
clean_tekst=[]
for i in range(0,len(deweynr)):
    tokens = tokenizer.tokenize(tekst[i])
    tekst_filtered=list()
    for k in range(len(tokens)):
        tokens[k]=tokens[k].lower()
    filtered_words = [word for word in tokens if word not in set(stopwords.words('norwegian'))]

    filtered_words = [x for x in filtered_words if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]


    norStem = snowball.NorwegianStemmer()
    stemmed_words = list()
    for word in filtered_words:
        stemmed_words.append(norStem.stem(word))

    tekst_filtered = ' '.join(stemmed_words)
    clean_tekst.append(tekst_filtered)
clean_tekst_pickle = open('clean_unique_text_stemmed_unique_new.pckl','wb')
tekst_og_labels=(clean_tekst, deweynr, titler)
pickle.dump(tekst_og_labels, clean_tekst_pickle)
clean_tekst_pickle.close()

### Vektorisering av tekst v.h.a tfidf.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# vect = TfidfVectorizer(max_df = 0.95, min_df = 0.05, norm='l2')
# X_idf = vect.fit_transform(clean_tekst)
# print(X_idf.shape)

# X = corpus_word2vec
# #print(X.shape)
# lb = LabelBinarizer()
# Y = lb.fit_transform(deweynr)
# #print('-------------')
# print(Y.shape)
# #print('-------------')
# #print(np.sum(Y, axis = 0))
# textual_features = (X_idf,Y, deweynr)
#
# feature_pickle = open('textual_features_idfreduced_stemmed_l2_knut0708_u10.pckl','wb')
# pickle.dump(textual_features, feature_pickle)
# feature_pickle.close()
end = time.time()
print(end - start)