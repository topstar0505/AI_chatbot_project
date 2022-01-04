import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    
    with open("data.pickle", "rb") as f:
        words, labels, train, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent["tag"]) #for classifying

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #convert all word to lower case and remove ?
    words = sorted(list(set(words))) #remove all duplicate and got a sorted list

    labels = sorted(labels)

    train = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        word = [stemmer.stem(w.lower()) for w in doc]

        for w in words: #going through all the stem word
            if w in word: #finding if word exist in current pattern
                bag.append(1) #if exists
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        train.append(bag)
        output.append(output_row)


    train = np.array(train)
    output = np.array(output)
    #above 2 array with be array consists of 0,1
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, train, output), f)

#model building
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(train, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sen in s_words:
        for x,w in enumerate(words):
            if w == sen:
                bag[x] = 1
    
    return np.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while(1):
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if (result[result_index] > 0.85):
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("I didn't get that, try again.")
chat()