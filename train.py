#! /usr/bin/python3

import sys
from contextlib import redirect_stdout

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Lambda

from dataset import *
from codemaps import *
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

USE_PREFIX = True   # prefix embedding
USE_CAPITAL = True  # capital letter embedding
USE_PROJECTION = False  
USE_CNN = False
def build_network(codes) :

   inputs = []
   embeddings = []
   # sizes
   n_words = codes.get_n_words()
   n_sufs = codes.get_n_sufs()
   n_prefs = codes.get_n_prefs()
   n_caps = codes.get_n_caps()
   n_labels = codes.get_n_labels()   
   max_len = codes.maxlen

   inptW = Input(shape=(max_len,)) # word input layer & embeddings
   embW = Embedding(input_dim=n_words, output_dim=100,
                    input_length=max_len)(inptW)  
   
   inptS = Input(shape=(max_len,))  # suf input layer & embeddings
   embS = Embedding(input_dim=n_sufs, output_dim=50,
                    input_length=max_len)(inptS) 

   inputs.append(inptW)
   embeddings.append(Dropout(0.1)(embW))
   inputs.append(inptS)
   embeddings.append(Dropout(0.1)(embS))

      # Conditional: prefix
   if USE_PREFIX:
      print("prefix")
      inptP = Input(shape=(max_len,))
      embP  = Embedding(input_dim=n_prefs, output_dim=50, input_length=max_len)(inptP)
      inputs.append(inptP)
      embeddings.append(Dropout(0.1)(embP))

   # Conditional: capitalization
   if USE_CAPITAL:
      print("capital")
      inptC = Input(shape=(max_len,))
      embC  = Embedding(input_dim=n_caps, output_dim=5, input_length=max_len)(inptC)
      inputs.append(inptC)
      embeddings.append(Dropout(0.1)(embC))

   drops = concatenate(embeddings) # concatenate all embeddings

   if USE_PROJECTION:
      print("projection")
      drops = TimeDistributed(Dense(128, activation='relu'), name='projection')(drops)

   if USE_CNN:
      print("cnn")
      conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='cnn')(drops)
      drops = concatenate([drops, conv])

   # biLSTM   
   bilstm = Bidirectional(LSTM(units=200, return_sequences=True,
                               recurrent_dropout=0.1))(drops) 
   # output softmax layer
   out = TimeDistributed(Dense(n_labels, activation="softmax"))(bilstm)

   # build and compile model
   model = Model(inputs=inputs, outputs=out) 
   model.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
   
   return model
   


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 5
pref_len = 3
codes  = Codemaps(traindata, max_len, suf_len, pref_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr) :
   model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

Xtrain_inputs = Xt[:2]  # word + suffix are always there

if USE_PREFIX:
    Xtrain_inputs.append(Xt[2])
if USE_CAPITAL:
    Xtrain_inputs.append(Xt[3])

# train model
with redirect_stdout(sys.stderr) :
   model.fit(Xtrain_inputs, Yt, batch_size=32, epochs=10, validation_data=(Xv,Yv), verbose=1)

# save model and indexs
model.save(modelname + ".keras")
codes.save(modelname + ".keras")

