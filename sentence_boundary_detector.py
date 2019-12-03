# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:11:23 2019

@author: Ivan Garrido Marquez
"""
try:
   import cPickle as pickle
except:
   import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import nltk
import numpy as np
import tools_for_SBD as sbdt

#This class will template a sentence boundary detector using a trained sklearn classifier
class sentenceBoundaryDetector:
    #Class constructor
    def __init__(self,classifier,context_size):
        #Load trained classifier
        class_file=open(classifier,'rb')
        self.sbd=pickle.load(class_file)
        self.k=context_size

    def detectSentencesBoundariesTest(self,doc):
        #Output sentences list
        out_sents=[]
        out_positions=[]
        #Input text document
        #Tokenize and POS tag the input document
        #text=nltk.tokenize.word_tokenize(doc)
        #pos_tokens=nltk.pos_tag(text,tagset='universal')
        pos_tokens=doc
        #Analyze each token with the classifier to decide if they are sentence boundaries or not
        #control variable to track the token
        i=0
        tmp_sentence=""
        for tkn in pos_tokens:
            #Each output sentence is formed by accumulating the tokens up to detecting a boundary
            #print(tkn[0].split('/')[0])
            #print(tkn)
            tmp_sentence+=tkn[0]+' '
            #Map the token to the vector space of features of its context
            vectorTkn=sbdt.vectorizeTokenWithPOSContext(i,self.k,pos_tokens)
            #Detect the boundaries by applying the classifier to token
            prediction=self.sbd.predict(vectorTkn.reshape(1,-1))
            #print(tkn[0].split('/')[0]+"<"+str(prediction)+">")
            #print(vectorTkn)
            if prediction==[1]:
                out_sents.append(tmp_sentence[:-1])
                out_positions.append(i)
                tmp_sentence=""
            i+=1
        out_sents.append(tmp_sentence[:-1])
        return out_sents,out_positions

    def detectSentencesBoundaries(self,doc):
        #Output sentences list
        out_sents=[]
        out_positions=[]
        #Input text document
        #Tokenize and POS tag the input document
        text=nltk.tokenize.word_tokenize(doc)
        pos_tokens=nltk.pos_tag(text,tagset='universal')
        #Analyze each token with the classifier to decide if they are sentence boundaries or not
        #control variable to track the token
        i=0
        tmp_sentence=""
        for tkn in pos_tokens:
            #Each output sentence is formed by accumulating the tokens up to detecting a boundary
            #print(tkn[0].split('/')[0])            
            tmp_sentence+=tkn[0].split('/')[0]+' '
            #Map the token to the vector space of features of its context
            vectorTkn=sbdt.vectorizeTokenWithPOSContext(i,self.k,pos_tokens)
            #Detect the boundaries by applying the classifier to token
            prediction=self.sbd.predict(vectorTkn.reshape(1,-1))
            #print(tkn[0].split('/')[0]+"<"+str(prediction)+">")
            #print(vectorTkn)
            if prediction==[1]:
                out_sents.append(tmp_sentence[:-1])
                out_positions.append(i)
                tmp_sentence=""
            i+=1
        out_sents.append(tmp_sentence[:-1])
        return out_sents,out_positions
    #Return all the found sentences