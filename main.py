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
docs = []

for data in data_file["data"]:
    for pattern in data["patterns"]:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs.append(pattern)

    if data["tag"] not in labels:
        labels.append(data["tag"])

