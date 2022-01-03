import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("data.json") as file:
    data_file = json.load(file)

words = []
labels = []
docs_x = [] #list of diff pattern
docs_y = [] #entry for docs_x

for data in data_file["data"]:  #loading json file
    for pattern in data["patterns"]:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs_x.append(pattern)
        docs_y.append(data["tag"]) #for classifying

    if data["tag"] not in labels:
        labels.append(data["tag"])

words = [stemmer.stem(w.lower()) for w in words] #convert all word to lower case
words = sorted(list(set(words))) #remove all duplicate and got a sorted list

labels = sorted(labels)

#making a bag for words
