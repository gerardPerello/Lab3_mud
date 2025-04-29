#!/usr/bin/python3

import sys
from contextlib import redirect_stdout

import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Embedding, Dropout, concatenate,
    Bidirectional, LSTM, TimeDistributed, Dense, Conv1D
)

from dataset import Dataset
from codemaps import Codemaps

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ——————————————————————————————————————————————
# FLAGS TO ENABLE/DISABLE FEATURES
USE_PREFIX      = True
USE_CAPITAL     = True
USE_SHAPE       = True
USE_LENGTH      = True
USE_DIGIT_FLAG  = True
USE_DASH_FLAG   = True
USE_PUNCT_FLAG  = True
USE_DRUGBANK    = True
USE_HSDB        = True
USE_PROJECTION  = False
USE_CNN         = False
# ——————————————————————————————————————————————

def build_network(codes):
    """
    Build a sequence-labeling model with togglable features:
    prefix, capitalization, shape, length, digit, dash,
    punctuation, DrugBank membership, HSDB membership.
    """
    inputs, embeddings = [], []

    # retrieve vocab sizes and max sequence length
    max_len   = codes.maxlen
    n_words   = codes.get_n_words()
    n_sufs    = codes.get_n_sufs()
    n_prefs   = codes.get_n_prefs()
    n_caps    = codes.get_n_caps()
    n_shapes  = codes.get_n_shapes()
    n_length  = codes.get_n_length()
    n_bool    = codes.get_n_bool()
    n_labels  = codes.get_n_labels()

    # 1) WORD input
    in_w  = Input(shape=(max_len,), name='word_in')
    emb_w = Embedding(n_words, 150, input_length=max_len)(in_w)
    inputs.append(in_w)
    embeddings.append(Dropout(0.1)(emb_w))

    # 2) SUFFIX input
    in_s  = Input(shape=(max_len,), name='suf_in')
    emb_s = Embedding(n_sufs, 50, input_length=max_len)(in_s)
    inputs.append(in_s)
    embeddings.append(Dropout(0.1)(emb_s))

    # 3) PREFIX input (optional)
    if USE_PREFIX:
        in_p  = Input(shape=(max_len,), name='pref_in')
        emb_p = Embedding(n_prefs, 50, input_length=max_len)(in_p)
        inputs.append(in_p)
        embeddings.append(Dropout(0.1)(emb_p))

    # 4) CAPITALIZATION input (optional)
    if USE_CAPITAL:
        in_c  = Input(shape=(max_len,), name='cap_in')
        emb_c = Embedding(n_caps, 5, input_length=max_len)(in_c)
        inputs.append(in_c)
        embeddings.append(Dropout(0.1)(emb_c))

    # 5) SHAPE input (optional)
    if USE_SHAPE:
        in_sh  = Input(shape=(max_len,), name='shape_in')
        emb_sh = Embedding(n_shapes, 25, input_length=max_len)(in_sh)
        inputs.append(in_sh)
        embeddings.append(Dropout(0.1)(emb_sh))

    # 6) LENGTH input (optional)
    if USE_LENGTH:
        in_len  = Input(shape=(max_len,), name='length_in')
        emb_len = Embedding(n_length, 5, input_length=max_len)(in_len)
        inputs.append(in_len)
        embeddings.append(Dropout(0.1)(emb_len))

    # 7) DIGIT-PRESENCE input (optional)
    if USE_DIGIT_FLAG:
        in_dig  = Input(shape=(max_len,), name='digit_in')
        emb_dig = Embedding(n_bool, 5, input_length=max_len)(in_dig)
        inputs.append(in_dig)
        embeddings.append(Dropout(0.1)(emb_dig))

    # 8) DASH-PRESENCE input (optional)
    if USE_DASH_FLAG:
        in_dash  = Input(shape=(max_len,), name='dash_in')
        emb_dash = Embedding(n_bool, 5, input_length=max_len)(in_dash)
        inputs.append(in_dash)
        embeddings.append(Dropout(0.1)(emb_dash))

    # 9) PUNCT-PRESENCE input (optional)
    if USE_PUNCT_FLAG:
        in_pun  = Input(shape=(max_len,), name='punct_in')
        emb_pun = Embedding(n_bool, 5, input_length=max_len)(in_pun)
        inputs.append(in_pun)
        embeddings.append(Dropout(0.1)(emb_pun))

    # 10) DRUGBANK membership input (optional)
    if USE_DRUGBANK:
        in_db  = Input(shape=(max_len,), name='db_in')
        emb_db = Embedding(n_bool, 5, input_length=max_len)(in_db)
        inputs.append(in_db)
        embeddings.append(Dropout(0.1)(emb_db))

    # 11) HSDB membership input (optional)
    if USE_HSDB:
        in_hs  = Input(shape=(max_len,), name='hsdb_in')
        emb_hs = Embedding(n_bool, 5, input_length=max_len)(in_hs)
        inputs.append(in_hs)
        embeddings.append(Dropout(0.1)(emb_hs))

    # concatenate all embeddings
    x = concatenate(embeddings)

    # optional projection and CNN
    if USE_PROJECTION:
        x = TimeDistributed(Dense(128, activation='relu'), name='projection')(x)
    if USE_CNN:
        conv = Conv1D(64, 3, padding='same', activation='relu', name='cnn')(x)
        x = concatenate([x, conv])

    # bi-LSTM + softmax output
    x   = Bidirectional(LSTM(200, return_sequences=True))(x)
    out = TimeDistributed(Dense(n_labels, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # usage: train.py <train_dir> <dev_dir> <model_name>
    traindir, valdir, modelname = sys.argv[1:4]

    # load datasets
    train_ds = Dataset(traindir)
    dev_ds   = Dataset(valdir)

    # set sequence and affix lengths
    max_len  = 150
    suf_len  = 5
    pref_len = 3
    codes    = Codemaps(train_ds, max_len, suf_len, pref_len)

    # build and summarize model
    model = build_network(codes)
    with redirect_stdout(sys.stderr):
        model.summary()

    # encode inputs and labels
    X_train_all = codes.encode_words(train_ds)  # list of up to 11 arrays
    Y_train     = codes.encode_labels(train_ds)
    X_val_all   = codes.encode_words(dev_ds)
    Y_val       = codes.encode_labels(dev_ds)

    # assemble inputs according to enabled features
    X_train = [X_train_all[0], X_train_all[1]]
    X_val   = [X_val_all[0],   X_val_all[1]]

    idx = 2
    if USE_PREFIX:      X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_CAPITAL:     X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_SHAPE:       X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_LENGTH:      X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_DIGIT_FLAG:  X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_DASH_FLAG:   X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_PUNCT_FLAG:  X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_DRUGBANK:    X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1
    if USE_HSDB:        X_train.append(X_train_all[idx]); X_val.append(X_val_all[idx]); idx += 1

    # train model
    with redirect_stdout(sys.stderr):
        model.fit(
            X_train, Y_train,
            batch_size=32, epochs=10,
            validation_data=(X_val, Y_val), verbose=1
        )

    # save model and codemaps
    model.save(modelname + '.keras')
    codes.save(modelname + '.keras')
