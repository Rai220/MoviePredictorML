import os
import re
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import tqdm
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout, Concatenate, LSTM
from keras.models import Model

STRING_SIZE = 150
letters = "@ABCDEFGHIJKLMNOPQRSTUVWXYZabccdefghijklmnopqrstuvwxyz0123456789.,!?-)(:/\" "
dictSize = len(letters)

def sparse(n, size):
    out = [0.0] * size
    if int(n) >= size:
        print("{} {}".format(n, size))
    out[int(n)] = 1.0
    return out

def chartoindex(c):
    #c = c.upper()
    if (c not in letters):
        print("Incorrect letter: " + c)
        return 0
    return letters.index(c)

def word2input(word):
    word = re.sub('[^0-9a-zA-Z.,!? \-)(\:/\"]+', '', word)
    input = list(map(lambda c: sparse(chartoindex(c), dictSize), word))
    input += [sparse(dictSize - 1, dictSize)] * (STRING_SIZE - len(input))
    input += [[len(word)] * dictSize] * 1
    return list(input)


def clean(text):
    text = text.strip()
    text = re.sub('[^0-9a-zA-Z.,!? \-)(\:]+', '', text)
    if (len(text) >= STRING_SIZE):
        text = text[:STRING_SIZE - 1]
    text = text.ljust(STRING_SIZE)
    return text

train = pd.read_csv("train.csv")

print("Train size: " + str(len(train)))

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

allText = ""
for st in list_sentences_train:
    allText += clean(st)

vocab = sorted(set(allText))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))


def getInput(data, start=0, step=0):
    input = list()

    if start == 0 and step == 0:
        dataToLoad = data
    else:
        stop = min(start+step, len(data))
        print("Saving data from line: " + str(start) + " to line: " + str(stop))
        dataToLoad = data[start:stop]

    for text in tqdm.tqdm(dataToLoad):
        input.append(word2input(clean(text)))


    inpArr = np.array(input)
    return inpArr

inp = Input(shape=(STRING_SIZE + 1, dictSize), name='main_input')
x = Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu',
           use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None)(inp)

x = MaxPooling1D(pool_size=4)(x)
x = Flatten()(x)
x = Dropout(0.5, noise_shape=None, seed=None)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5, noise_shape=None, seed=None)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(input=[inp], outputs=x)

def getPatch(train):
    print("Get big patch")
    #train = shuffle(train)
    #subTrain = train.head(30000)
    subTrain = train

    input = getInput(subTrain["comment_text"].fillna("_na_").values)
    y = subTrain[list_classes].values

    return input, y

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training...")
x_train, y_train = getPatch(train)
#for i in range(0, 10):

model.fit(x_train, y_train, batch_size=1000, epochs=2)

test = pd.read_csv("test.csv")
list_sentences_test = test["comment_text"].fillna("_na_").values
print("Test shape: " + str(list_sentences_test.shape))
step = 10000
start = 0
y_result = ""
while (start <= len(list_sentences_test)):
    input_test = getInput(list_sentences_test, start, step)
    y_test = model.predict(input_test, batch_size=100, verbose=1)
    if y_result == "":
        y_result = y_test
    else:
        y_result = np.concatenate((y_result, y_test), axis=0)
    start += step

print("y_result size: " + str(y_result.shape))

if os._exists('conv.csv'):
    os.remove('conv.csv')

sample_submission = pd.read_csv(f'sample_submission.csv')
sample_submission[list_classes] = y_result
sample_submission.to_csv('conv.csv', index=False)

test = pd.read_csv("train.csv")
list_sentences_test = test["comment_text"].fillna("_na_").values
print("Test shape: " + str(list_sentences_test.shape))
step = 10000
start = 0
y_result = ""
while (start <= len(list_sentences_test)):
    input_test = getInput(list_sentences_test, start, step)
    y_test = model.predict(input_test, batch_size=100, verbose=1)
    if y_result == "":
        y_result = y_test
    else:
        y_result = np.concatenate((y_result, y_test), axis=0)
    start += step

print("y_result size: " + str(y_result.shape))

if os._exists('conv_train.csv'):
    os.remove('conv_train.csv')

sample_submission = pd.read_csv(f'train.csv')
sample_submission.drop("comment_text", axis=1, inplace=True)
sample_submission[list_classes] = y_result
sample_submission.to_csv('conv_train.csv', index=False)