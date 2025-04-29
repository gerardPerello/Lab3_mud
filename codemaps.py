
import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, preflen=None):

        if isinstance(data,Dataset) and maxlen is not None and suflen is not None and preflen is not None:
            self.__create_indexs(data, maxlen, suflen, preflen)

        elif type(data) == str and maxlen is None and suflen is None and preflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, preflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = preflen
        words = set([])
        lc_words = set([])
        sufs = set([])
        labels = set([])
        prefs = set([])
        caps = set([])
        
        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                sufs.add(t['lc_form'][-self.suflen:])
                labels.add(t['tag'])
                prefs.add(t['lc_form'][:self.preflen])
                w = t['form']
                if   w.istitle(): caps.add('TITLE')
                elif w.isupper(): caps.add('UPPER')
                else:             caps.add('LOWER')

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding

        self.pref_index  = {p:i+2 for i,p in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0  # Padding
        self.pref_index['UNK'] = 1  # Unknown prefixes

        self.cap_index = {c:i+2 for i,c in enumerate(list(caps))}
        self.cap_index['PAD'] = 0
        self.cap_index['UNK'] = 1

        
        print(prefs)
        print("\n")
        print(sufs)
        print(caps)

    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.cap_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                elif t == 'PREF': self.pref_index[k] = int(i)
                elif t == 'CAP': self.cap_index[k] = int(i)
                elif t == 'PREFLEN': self.preflen = int(k)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('SUFLEN', self.preflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)
            for key in self.cap_index : print('CAP', key, self.cap_index[key], file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])
        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])
        Xp = [[ self.pref_index.get(tok['lc_form'][:self.preflen], self.pref_index['UNK'])
        for tok in sent ] for sent in data.sentences()]
        Xp = pad_sequences(Xp, maxlen=self.maxlen, padding='post', value=self.pref_index['PAD'])
        # return encoded sequences
        # --- capitalization
        def cap_cat(form):
            if form.istitle(): return 'TITLE'
            elif form.isupper(): return 'UPPER'
            else: return 'LOWER'
        Xc = [[ self.cap_index.get(cap_cat(tok['form']), self.cap_index['UNK'])
                for tok in sent ] for sent in data.sentences()]
        Xc = pad_sequences(Xc, maxlen=self.maxlen, padding='post', value=self.cap_index['PAD'])

        return [ Xw, Xs, Xp, Xc ]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    ## -------- get n pref ---------
    def get_n_prefs(self): 
        return len(self.pref_index)
    ## -------- get cap index size ---------
    def get_n_caps(self): 
        return len(self.cap_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

