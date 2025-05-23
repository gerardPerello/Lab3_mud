#! /usr/bin/python3

import sys
from os import system

from tensorflow.keras.models import Model,load_model

from dataset import *
from codemaps import *
import evaluator

## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def output_entities(data, preds, outfile) :

   outf = open(outfile, 'w')
   for sid,tags in zip(data.sentence_ids(),preds) : 
      inside = False
      for k in range(0,min(len(data.get_sentence(sid)),codes.maxlen)) :
         y = tags[k]
         token = data.get_sentence(sid)[k]
            
         if (y[0]=="B") :
             entity_form = token['form']
             entity_start = token['start']
             entity_end = token['end']
             entity_type = y[2:]
             inside = True
         elif (y[0]=="I" and inside) :
             entity_form += " "+token['form']
             entity_end = token['end']
         elif (y[0]=="O" and inside) :
             print(sid, str(entity_start)+"-"+str(entity_end), entity_form, entity_type, sep="|", file=outf)
             inside = False
        
      if inside : print(sid, str(entity_start)+"-"+str(entity_end), entity_form, entity_type, sep="|", file=outf)
            
   outf.close()

## --------- Evaluator ----------- 
def evaluation(datadir,outfile) :
   evaluator.evaluate("NER", datadir, outfile)

   
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

fname = sys.argv[1]
datadir = sys.argv[2]
outfile = sys.argv[3]

model = load_model(fname)
codes = Codemaps(fname)

testdata = Dataset(datadir)
X_all = codes.encode_words(testdata)

num_inputs = len(model.inputs)
X_inputs = X_all[:num_inputs]

Y_probs = model.predict(X_inputs)
Y_tags = [
   [codes.idx2label(np.argmax(w)) for w in seq]
   for seq in Y_probs
]

# extract & evaluate entities with basic model
output_entities(testdata, Y_tags, outfile)
evaluation(datadir,outfile)

