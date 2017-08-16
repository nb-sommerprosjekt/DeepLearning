import os
import pymarc
import pickle
import fasttext
import gensim

file_name = "all_dewey_dump.xml"
full_file = os.path.abspath(os.path.join('dewey_dump',file_name))


#nyttig error:  https://github.com/edsu/pymarc/issues/73
klassebetegnelser = []
deweynr = []
with open(full_file, 'rb') as fh:
    for record in pymarc.parse_xml_to_array(fh):
        if "153" in record:
            if "j" in record["153"] and "a" in record["153"]:
                klassebetegnelser.append(record['153']['j'])
                deweynr.append(record['153']['a'])

zipped = list(zip(deweynr,klassebetegnelser))

#removing punctuations in deweynr
zipped_wop = [tuple(s.replace(".","") for s in tup) for tup in zipped]


# Loading pickle with deweynr from our db
with open('tekst_label_all_knut0708_no_unique.pckl', "rb") as openfile:
    tekst_labels = pickle.load(openfile)

deweynr_db = tekst_labels[0]
unique_dewey = list()
for i in deweynr_db:
    if i not in unique_dewey:
        unique_dewey.append(i)
print (len(unique_dewey))
print(len(deweynr_db))

#Use Gensim, load fastText trained .vec file with load.word2vec models and use most_similiar() method to find similar words!
wordvec = gensim.models.wrappers.FastText.load_word2vec_format('wiki.no/wiki.no.vec')
print(wordvec['innhøsting'])