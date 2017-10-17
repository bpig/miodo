# coding:utf-8
# 2017/10/17 下午4:08

# http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl

import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
print "finish load data"

trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

# converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

net = tflearn.input_data([None, 100])

net1 = tflearn.embedding(net, input_dim=10000, output_dim=128)
net1 = tflearn.lstm(net1, 128, dropout=0.5)
# net = tflearn.fully_connected(net1, 2, activation='softmax')

net2 = tflearn.fully_connected(net, 128, activation="relu")
# net = tflearn.fully_connected(net2, 2, activation="softmax")

net = tflearn.merge([net1, net2], mode='concat')
net = tflearn.fully_connected(net, 2, activation="softmax")

net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=False, batch_size=32, n_epoch=10)

# lstm+dnn
# Training Step: 704  | total loss: 4.15139 | time: 87.095s
# | Adam | epoch: 001 | loss: 4.15139 | val_loss: 4.60098 -- iter: 22500/22500

# lstm
# Training Step: 704  | total loss: 10.80964 | time: 2.720s
# | Adam | epoch: 001 | loss: 10.80964 | val_loss: 11.10966 -- iter: 22500/22500

# dnn
# Training Step: 704  | total loss: 3.67918 | time: 2.661s
# | Adam | epoch: 001 | loss: 3.67918 | val_loss: 3.71019 -- iter: 22500/22500

# lstm 10
# Training Step: 7040  | total loss: 0.20675 | time: 87.094s
# | Adam | epoch: 010 | loss: 0.20675 | val_loss: 0.44841 -- iter: 22500/22500

# dnn+lstm 10
# Training Step: 7040  | total loss: 0.14053 | time: 91.280s
# | Adam | epoch: 010 | loss: 0.14053 | val_loss: 0.59019 -- iter: 22500/22500
if __name__ == "__main__":
    pass
