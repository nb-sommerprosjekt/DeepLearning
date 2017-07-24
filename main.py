from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import NorwegianStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Leser inn pickle med label + tittel laget som dict med label som key og tittel som value.
#title_labels = []
with (open("title_label_all.pkl", "rb")) as openfile:
        title_labels=pickle.load(openfile)

# Leser inn pickle med label + tekst laget som dict med label som key og tekst som value.
tekst_labels = []
with (open("tekst_label_all.pkl", "rb")) as openfile:
    tekst_labels = pickle.load(openfile)


# Natural Language Processing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Cleaning the texts

corpus = []
label = []

for key,value in tekst_labels.items():
    #print (value)
    value =re.sub('[^ÆØÅæøåa-zA-z]',' ',value)
    #print(value)
    value = value.lower()
    #print(value)
    value = value.split()
    #print(value)
    NorStem = NorwegianStemmer()
    value = [NorStem.stem(word) for word in value if not word in set(stopwords.words('norwegian'))]
    #print(value)
    value = ' '.join(value)
    #print(value)
    corpus.append(value)
    label.append(key)

print(label)

# Creating the Bag of Words model

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
#y = label.loc[:].values
#print(y)
#
# print(X)
# print(y)
#
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(corpus, label, test_size = 0.20, random_state = 0)

#
#
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print (cm)
# accuracy = (55+91)/200
# print(accuracy)