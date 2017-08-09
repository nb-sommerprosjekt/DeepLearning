import os
import re
import pickle
#Lager pickles som inneholder tittel/tekst og dewey i formatet [dewey][tittel/tekst]]
def text_to_pickle():
    rootdir = "/home/ubuntu/Downloads/knut_0708/none_under_ten"

    label_list=list()
    text_list = list()
    title_list =list()
    pickle_title = open('title_label_knut0708.pckl', 'ab+')
    pickle_tekst = open('tekst_label_all_knut0708.pckl', 'ab+')
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if str(file)[:5] == "meta-":
                f = open(os.path.join(subdir, file), "r+")
                meta_tekst = f.read()

                dewey = re.search('dewey:::(.+?)\.', meta_tekst)
                if dewey:
                    found_dewey = dewey.group(1)

                title = re.search('tittel:::(.+?)\n', meta_tekst)
                if title:
                    found_title = title.group(1)

                file_name=os.path.join(subdir, file)
                file_name_text=file_name.replace("meta-","")
                tekst_fil=open(os.path.join(subdir, file_name_text), "r+")
                tekst=tekst_fil.read()
                tekst_fil.close()
                label_list.append(found_dewey)
                text_list.append(tekst)
                title_list.append(found_title)

    pickle.dump([label_list,text_list], pickle_tekst)
    pickle.dump([label_list, title_list], pickle_title)
    pickle_tekst.close()
    pickle_title.close()

text_to_pickle()