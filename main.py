import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
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
        docs_x.append(word)
        docs_y.append(data["tag"]) #for classifying

    if data["tag"] not in labels:
        labels.append(data["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #convert all word to lower case and remove ?
words = sorted(list(set(words))) #remove all duplicate and got a sorted list

labels = sorted(labels)

#making a bag for words for determine occurence of words
train = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag = []

    word = [stemmer.stem(w) for w in doc] #stem the words

    #going through all the stem word
    for w in words:
        if w in word: #finding if word exist in current pattern
            bag.append(1) #if exists
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    train.append([bag])
    output.append(output_row)

train = np.array(train)
output = np.array(output)
#above 2 array with be array consists of 0,1

#building model
