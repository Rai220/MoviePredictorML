import os, sys
import re

import numpy as np
import pandas as pd
import tqdm
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.shape)
print(df_test.shape)

# combine the train and test sets for encoding and padding

train_len = len(df_train)
df_combined =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)

print(df_combined.shape)

# define text data
docs_combined = df_combined['comment_text'].astype(str)

# initialize the tokenizer
t = Tokenizer()
t.fit_on_texts(docs_combined)
vocab_size = len(t.word_index) + 1

print("Vocab size: " + str(vocab_size))

# integer encode the text data
encoded_docs = t.texts_to_sequences(docs_combined)

# pad the vectors to create uniform length
padded_docs_combined = pad_sequences(encoded_docs, maxlen=500, padding='post')

sys.exit(0)



STRING_SIZE = 100
letters = "@ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? -)(:"
dictSize = len(letters)

def sparse(n, size):
    out = [0.0] * size
    if int(n) >= size:
        print("{} {}".format(n, size))
    out[int(n)] = 1.0
    return out

def chartoindex(c):
    c = c.upper()
    if (c not in letters):
        print("Incorrect letter: " + c)
        return 0
    return letters.index(c)

def word2input(word):
    word = word.upper()
    word = re.sub('[^0-9A-Z.,!? \-)(\:]+', '', word)
    input = list(map(lambda c: sparse(chartoindex(c), dictSize), word.upper().replace(" ", "")))
    input += [sparse(dictSize - 1, dictSize)] * (STRING_SIZE - len(input))
    return list(input)

def clean(text):
    text = text.strip().upper()
    text = re.sub('[^0-9A-ZА-Я.,!?\-)(\'\"\]\[\: ]+', '', text)
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


def getInput(data, aug=False):
    input = list()

    for text in tqdm.tqdm(data):
        #if (aug):
        #    words = text.split()
        #    r = randint(0, 6)
        #    if (r == 1):
        #        words.insert(randint(0,len(words)), choice(stopWords))
        #    if (r == 2):
        #        words.insert(randint(0,len(words)), choice(stopWords))
        #        words.insert(randint(0,len(words)), choice(stopWords))
        #    text = " ".join(words)

        input.append(word2input(clean(text)))

    print("Saving to array...")
    inpArr = np.array(input)
    #inpArr = inpArr.astype('float32')
    return inpArr

inp = Input(shape=(STRING_SIZE, dictSize), name='main_input')
x = Conv1D(200, 7, strides=1, padding='same', dilation_rate=1, activation='relu',
           use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
           kernel_constraint=None, bias_constraint=None)(inp)

x = MaxPooling1D(pool_size=4)(x)
x = Flatten()(x)
x = Dropout(0.8, noise_shape=None, seed=None)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.8, noise_shape=None, seed=None)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(input=[inp], outputs=x)

def getPatch(train):
    print("Get big patch")
    #train = shuffle(train)
    #subTrain = train.head(10000)
    subTrain = train

    input = getInput(subTrain["comment_text"].fillna("_na_").values, aug=True)
    y = subTrain[list_classes].values

    return input, y

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training...")
x_train, y_train = getPatch(train)
for i in range(0, 10):
    model.fit(x_train, y_train, batch_size=1000, epochs=5)


    test = pd.read_csv("test.csv")
    list_sentences_test = test["comment_text"].fillna("_na_").values
    input_test = getInput(list_sentences_test)
    y_test = model.predict(input_test, batch_size=1000, verbose=1)

    if os._exists('submission_' + str(i) + '.csv'):
        os.remove('submission_' + str(i) + '.csv')

    sample_submission = pd.read_csv(f'sample_submission.csv')
    sample_submission[list_classes] = y_test
    sample_submission.to_csv('submission_' + str(i) + '.csv', index=False)