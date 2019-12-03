# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:29:22 2019

@author: Ivan Garrido Marquez
"""

import argparse
import tools_for_SBD as sbdt
from nltk.corpus import brown
import sentence_boundary_detector as sbd
import nltk

#print(brown.fileids())

sbd_obj=sbd.sentenceBoundaryDetector("classifiers/tree_classifier_model_SBD_training_set_matrix_k2_n50_i0_pickle_pickle.bin",2)
#sbd_obj=sbd.sentenceBoundaryDetector("classifiers/mlp_classifier_model_SBD_training_set_matrix_k2_n50_i0_pickle_pickle.bin",2)

#sentences=sbd_obj.detectSentencesBoundaries(brown.raw("cp02"))
#detectSentencesBoundariesTest
#text1=' '.join(nltk.corpus.brown.words("ca01"))
text1=' '.join(nltk.corpus.brown.words("cp02"))
#text=nltk.tokenize.word_tokenize(text1)
sentences,positionsInDoc=sbd_obj.detectSentencesBoundaries(text1)
#sentences,positionsInDoc=sbd_obj.detectSentencesBoundaries(brown.raw("ca01"))
#sentences,positionsInDoc=sbd_obj.detectSentencesBoundariesTest(brown.tagged_words("cp02",tagset='universal'))

num=1
for sent in sentences:
    print(str(num)+" - "+sent)
    num+=1
#
#boundsDetected=sbdt.getEOSIndicesOfDoc(sentences)
print(len(sentences))

#b=brown.sents("cp02")#,tagset='universal')
b=brown.sents("ca01")#,tagset='universal')
boundsDetected=sbdt.getEOSIndicesOfDoc(b)
print(len(b))
for s in b:
    print(s)
print(boundsDetected)
print(positionsInDoc)