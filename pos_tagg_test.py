# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:18:08 2019

@author: ivan
"""
import nltk
from nltk.corpus import brown

def mapTagsToUniversal(tokens):
    result=[]
    for x in tokens:
        result.append(nltk.tag.mapping.map_tag('brown','universal',x[1]))
    return result

brown_tagged_sents = brown.tagged_sents(brown.fileids()[:50])
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

ca01_txt=' '.join(nltk.corpus.brown.words('ca01'))

unitag=unigram_tagger.tag(nltk.tokenize.word_tokenize(ca01_txt))
#unitag=unigram_tagger.tag(nltk.tokenize.word_tokenize(ca01_txt),tagset='universal')

#print(unitag)
#print(nltk.pos_tag(unitag,tagset='universal'))

#print(mapTagsToUniversal(unitag))