# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:33:02 2019

@author: ivan
"""

import nltk
#from sets import Set
#nltk.download() # this opens a GUI to download all corpora needed
from nltk.corpus import brown

#sentences = brown.sents()
#print("Number of sentences")
#print(len(sentences))
#print("Number of documents")
#print(len(brown.fileids()))
corpus_ids=brown.fileids()
print(corpus_ids[:50])
print(brown.tagged_sents("cp02"))

def getEOSIndicesOfDoc(sents):
    indices=[]
    cursor=0
    for s in sents:
        cursor+=len(s)
        indices.append(cursor)
    return indices

def nextIndices(st,nd,current_end,indx):
    new_st=nd
    new_nd=indx[current_end+1]
    next_ind=current_end+1
    return new_st,new_nd,next_ind

#print(brown.words("cp02"))
#print(brown.tagged_words("cp02",tagset='universal'))
a=brown.tagged_sents("cp02",tagset='universal')[0]
print(a)
a1=brown.tagged_words("cp02",tagset='universal')[:len(a)]
print(a1)

b=brown.tagged_sents("cp02",tagset='universal')[1]
print(b)
b1=brown.tagged_words("cp02",tagset='universal')[len(a):len(a)+len(b)]
print(b1)

c=brown.tagged_sents("cp02",tagset='universal')[2]
print(c)
c1=brown.tagged_words("cp02",tagset='universal')[len(a)+len(b):len(a)+len(b)+len(c)]
print(c1)

if a==a1:
    print("A")

if b==b1:
    print("B")
    
if c==c1:
    print("C")

indx=getEOSIndicesOfDoc(brown.tagged_sents("cp02",tagset='universal'))
st=0
nd=indx[0]
x=0
print(brown.tagged_words("cp02",tagset='universal')[st:nd])
st,nd,x=nextIndices(st,nd,x,indx)
#st=nd
#nd=indx[1]
print(brown.tagged_words("cp02",tagset='universal')[st:nd])
st,nd,x=nextIndices(st,nd,x,indx)
#st=nd
#nd=indx[2]
print(brown.tagged_words("cp02",tagset='universal')[st:nd])
#print(nltk.pos_tag(sentences[0]))
#patterns=[]
#for x in sentences:
#    tokenlast=nltk.pos_tag(x,tagset='universal')[-1]
#    if tokenlast not in patterns:
#        print(tokenlast)
#        patterns.append(tokenlast)


#difPat=set(patterns)
#print(patterns)
    
