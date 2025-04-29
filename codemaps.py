import os
from dataset import Dataset
import numpy as np
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Codemaps:
    """
    Class to create and manage code maps (indices) for words, suffixes,
    prefixes, labels, capitalization, token shapes, length categories, boolean
    features (digit, dash, punctuation), and dictionary membership
    (DrugBank and HSDB) from a Dataset.
    """
    # Paths to external resources
    DRUGBANK_PATH = './resources/DrugBank.txt'
    HSDB_PATH     = './resources/HSDB.txt'
    def __init__(self, data, maxlen=None, suflen=None, preflen=None):
        # Load external dictionaries
        self.drugbank = {}
        self.hsdb = set()
        with open(self.DRUGBANK_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if "|" in line:
                    term, label = line.strip().split("|")
                    self.drugbank[term.lower()] = label.lower()
        with open(self.HSDB_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip().lower()
                if token:
                    self.hsdb.add(token)

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None and preflen is not None:
            self._create_indices(data, maxlen, suflen, preflen)
        elif isinstance(data, str) and maxlen is None and suflen is None and preflen is None:
            self._load_indices(data)
        else:
            raise ValueError('Codemaps: invalid or missing constructor parameters')

    def _create_indices(self, data, maxlen, suflen, preflen):
        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = preflen

        words, sufs, prefs, labels = set(), set(), set(), set()
        caps, shapes = set(), set()
        length_cats = ['short', 'medium', 'long']

        for sentence in data.sentences():
            for token in sentence:
                form = token['form']
                lc   = token['lc_form']
                lower = lc.lower()
                # basic
                words.add(form)
                sufs.add(lc[-self.suflen:])
                prefs.add(lc[:self.preflen])
                labels.add(token['tag'])
                # capitalization
                if form.istitle(): caps.add('TITLE')
                elif form.isupper(): caps.add('UPPER')
                else: caps.add('LOWER')
                # shape
                shapes.add(self._get_shape(form))

        # initialize indices
        self.word_index  = {'PAD':0, 'UNK':1}
        for i, w in enumerate(sorted(words), start=2): self.word_index[w] = i

        self.suf_index   = {'PAD':0, 'UNK':1}
        for i, s in enumerate(sorted(sufs), start=2): self.suf_index[s] = i

        self.pref_index  = {'PAD':0, 'UNK':1}
        for i, p in enumerate(sorted(prefs), start=2): self.pref_index[p] = i

        self.label_index = {'PAD':0}
        for i, t in enumerate(sorted(labels), start=1): self.label_index[t] = i

        self.cap_index   = {'PAD':0, 'UNK':1}
        for i, c in enumerate(sorted(caps), start=2): self.cap_index[c] = i

        self.shape_index = {'PAD':0, 'UNK':1}
        for i, sh in enumerate(sorted(shapes), start=2): self.shape_index[sh] = i

        self.length_index = {'PAD':0}
        for i, cat in enumerate(length_cats, start=1): self.length_index[cat] = i

        # boolean flags: pad, false, true
        self.bool_index   = {'PAD':0, 'FALSE':1, 'TRUE':2}

    def _load_indices(self, name):
        self.maxlen = self.suflen = self.preflen = 0
        # reset maps
        self.word_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.label_index = {}
        self.cap_index = {}
        self.shape_index = {}
        self.length_index = {}
        self.bool_index = {}

        with open(name + '.idx', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                tag = parts[0]
                if tag == 'MAXLEN':   self.maxlen = int(parts[1])
                elif tag == 'SUFLEN': self.suflen = int(parts[1])
                elif tag == 'PREFLEN':self.preflen = int(parts[1])
                elif tag == 'WORD':   self.word_index[parts[1]] = int(parts[2])
                elif tag == 'SUF':    self.suf_index[parts[1]] = int(parts[2])
                elif tag == 'PREF':   self.pref_index[parts[1]] = int(parts[2])
                elif tag == 'LABEL':  self.label_index[parts[1]] = int(parts[2])
                elif tag == 'CAP':    self.cap_index[parts[1]] = int(parts[2])
                elif tag == 'SHAPE':  self.shape_index[parts[1]] = int(parts[2])
                elif tag == 'LENGTH': self.length_index[parts[1]] = int(parts[2])
                elif tag == 'BOOL':   self.bool_index[parts[1]] = int(parts[2])

    def save(self, name):
        with open(name + '.idx', 'w') as f:
            print('MAXLEN', self.maxlen, '-', file=f)
            print('SUFLEN', self.suflen, '-', file=f)
            print('PREFLEN', self.preflen, '-', file=f)
            for key, val in sorted(self.label_index.items()): print('LABEL', key, val, file=f)
            for key, val in sorted(self.word_index.items()): print('WORD', key, val, file=f)
            for key, val in sorted(self.suf_index.items()):   print('SUF', key, val, file=f)
            for key, val in sorted(self.pref_index.items()):  print('PREF', key, val, file=f)
            for key, val in sorted(self.cap_index.items()):   print('CAP', key, val, file=f)
            for key, val in sorted(self.shape_index.items()): print('SHAPE', key, val, file=f)
            for key, val in sorted(self.length_index.items()):print('LENGTH', key, val, file=f)
            for key, val in sorted(self.bool_index.items()):  print('BOOL', key, val, file=f)

    def _get_shape(self, token):
        shape = []
        for ch in token:
            if ch.isupper():   shape.append('X')
            elif ch.islower(): shape.append('x')
            elif ch.isdigit(): shape.append('d')
            else:              shape.append(ch)
        return ''.join(shape)

    def _length_category(self, token):
        l = len(token)
        if l <= 4:    return 'short'
        elif l <= 8:  return 'medium'
        else:         return 'long'

    def encode_words(self, data):
        # word, suffix, prefix, capitalization
        Xw = [[self.word_index.get(tok['form'], self.word_index['UNK']) for tok in s] for s in data.sentences()]
        Xs = [[self.suf_index.get(tok['lc_form'][-self.suflen:], self.suf_index['UNK']) for tok in s] for s in data.sentences()]
        Xp = [[self.pref_index.get(tok['lc_form'][:self.preflen], self.pref_index['UNK']) for tok in s] for s in data.sentences()]
        Xc = [[self.cap_index.get('TITLE' if t['form'].istitle() else 'UPPER' if t['form'].isupper() else 'LOWER', self.cap_index['UNK']) for t in s] for s in data.sentences()]

        # shape
        Xsh = [[self.shape_index.get(self._get_shape(t['form']), self.shape_index['UNK']) for t in s] for s in data.sentences()]

        # length
        Xlen = [[self.length_index[self._length_category(t['form'])] for t in s] for s in data.sentences()]

        # boolean flags: digit, dash, punctuation
        Xdig  = [[self.bool_index['TRUE'] if any(ch.isdigit() for ch in t['form']) else self.bool_index['FALSE'] for t in s] for s in data.sentences()]
        Xdash = [[self.bool_index['TRUE'] if '-' in t['form'] else self.bool_index['FALSE'] for t in s] for s in data.sentences()]
        Xpun  = [[self.bool_index['TRUE'] if any(ch in string.punctuation for ch in t['form']) else self.bool_index['FALSE'] for t in s] for s in data.sentences()]

        # dictionary membership
        Xdb   = [[self.bool_index['TRUE'] if t['form'].lower() in self.drugbank else self.bool_index['FALSE'] for t in s] for s in data.sentences()]
        Xhsdb = [[self.bool_index['TRUE'] if t['form'].lower() in self.hsdb else self.bool_index['FALSE'] for t in s] for s in data.sentences()]

        # pad all
        arrays = [Xw, Xs, Xp, Xc, Xsh, Xlen, Xdig, Xdash, Xpun, Xdb, Xhsdb]
        padded = [pad_sequences(arr, maxlen=self.maxlen, padding='post', value=idx_map['PAD' if arr is None else None_id]) 
                  for arr, idx_map, None_id in zip(
                      arrays,
                      [self.word_index, self.suf_index, self.pref_index, self.cap_index,
                       self.shape_index, self.length_index, self.bool_index, self.bool_index,
                       self.bool_index, self.bool_index, self.bool_index],
                      ['PAD','PAD','PAD','PAD','PAD','PAD','PAD','PAD','PAD','PAD','PAD'])]
        return padded

    def encode_labels(self, data):
        Y = [[self.label_index.get(tok['tag'], self.label_index['PAD']) for tok in s] for s in data.sentences()]
        Y = pad_sequences(Y, maxlen=self.maxlen, padding='post', value=self.label_index['PAD'])
        return np.array(Y)

    # getters for sizes
    def get_n_words(self): return len(self.word_index)
    def get_n_sufs(self):  return len(self.suf_index)
    def get_n_prefs(self): return len(self.pref_index)
    def get_n_labels(self):return len(self.label_index)
    def get_n_caps(self):  return len(self.cap_index)
    def get_n_shapes(self):return len(self.shape_index)
    def get_n_length(self):return len(self.length_index)
    def get_n_bool(self):  return len(self.bool_index)

    # individual lookups
    def word2idx(self, w): return self.word_index.get(w, self.word_index['UNK'])
    def suf2idx(self, s):  return self.suf_index.get(s, self.suf_index['UNK'])
    def pref2idx(self,p):  return self.pref_index.get(p, self.pref_index['UNK'])
    def label2idx(self,l): return self.label_index.get(l, self.label_index['PAD'])
    def cap2idx(self,c):   return self.cap_index.get(c, self.cap_index['UNK'])
    def shape2idx(self,sh):return self.shape_index.get(sh, self.shape_index['UNK'])
    def length2idx(self,c):return self.length_index.get(c, self.length_index['PAD'])
    def bool2idx(self,b):  return self.bool_index.get(b, self.bool_index['PAD'])

    def idx2label(self, i):
        for k, v in self.label_index.items():
            if v == i: return k
        raise KeyError(f"Unknown label for index {i}")
