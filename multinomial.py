import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import time
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')


# Leser inn pickle med label + tittel laget som list av formatet [[labels][title]]
feature_pickle = open('textual_features_idfreduced_stemmed.pckl', 'rb')
textual_features = pickle.load(feature_pickle)
feature_pickle.close()

(X, Y,final_labels) = textual_features

Y=np.asarray(final_labels)




X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=1)

#---- Train model -----#
nb = MultinomialNB()
tid=time.time()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
nb.fit(X_train, y_train)
print(time.time()-tid)


y_pred_class = nb.predict(X_test)
print("Printing ")
print(metrics.accuracy_score(y_test, y_pred_class))


