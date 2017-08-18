import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import time
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')


feature_pickle = open('textual_features_idfreduced_stemmed_l2.pckl', 'rb')
textual_features = pickle.load(feature_pickle)
feature_pickle.close()

(X, Y,final_labels) = textual_features

Y=np.asarray(final_labels)



X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=1)

#---- Train model -----#
logreg = LogisticRegression()
tid=time.time()

logreg.fit(X_train, y_train)
print("Dette tok {} sekunder".format(time.time()-tid))
tid=time.time()
#scores = cross_val_score(logreg, X, Y, cv=5)
#print(scores)



y_pred_logreg_class = logreg.predict(X_test)
print("Printing accuracy for logress: ")
print(metrics.accuracy_score(y_test, y_pred_logreg_class))

