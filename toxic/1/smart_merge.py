import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.layers import Dense, Input, Dropout
from keras.models import Model



#allData.to_csv('allData.csv', index=False)

def loadData(f1, f2, f3):
    train = pd.read_csv(f1)
    train.drop("comment_text", axis=1, inplace=True)

    bn_fasttext_train_064 = pd.read_csv(f2)
    bn_fasttext_train_064.drop("id", axis=1, inplace=True)
    bn_fasttext_train_064.columns=["1", "2", "3", "4", "5", "6"]

    lstm = pd.read_csv(f3)
    lstm.drop("id", axis=1, inplace=True)
    lstm.columns=["7", "8", "9", "10", "11", "12"]

    print(train.shape)
    print(bn_fasttext_train_064.shape)
    print(lstm.shape)

    allData = pd.concat([train, bn_fasttext_train_064, lstm], axis=1)
    print(allData.shape)
    return allData

allData = loadData("train.csv", "submission_bn_fasttext_train_064.csv", "lstm_train_067.csv")


data_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
predict_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_x = allData[data_classes].values
train_y = allData[predict_classes].values


inp = Input(shape=(12,))
x = Dropout(0.5)(inp)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=256, epochs=25)

allData = loadData("test.csv", "submission_bn_fasttext_064.csv", "lstm_067.csv")
test_x = allData[data_classes].values
y_test = model.predict(test_x, batch_size=1024, verbose=1)

if os._exists('smart_merge.csv'):
    os.remove('smart_merge.csv')
sample_submission = pd.read_csv(f'sample_submission.csv')
sample_submission[predict_classes] = y_test
sample_submission.to_csv('smart_merge.csv', index=False)