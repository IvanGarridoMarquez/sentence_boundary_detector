# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:23:05 2019

@author: Ivan Garrido Marquez
"""

import argparse
import tools_for_SBD as sbd
from nltk.corpus import brown
import os
import nltk
#import scipy as sp
try:
   import cPickle as pickle
except:
   import pickle


def makeTrainingSet(training_docs,context_window_size):
    train_matrix=[]
    train_labels=[]
    
    for doc in training_docs:
        training_sentences=brown.tagged_sents(doc,tagset='universal')
        #sbd_indx=sbd.getEOSIndicesOfDoc(training_sentences)
        #text=brown.raw(doc)
        #text1=' '.join(nltk.corpus.brown.words(doc))
        #text=nltk.tokenize.word_tokenize(text1)
        text=nltk.corpus.brown.words(doc)
        full_doc=nltk.pos_tag(text,tagset='universal')
        tk_position=0
        for sentd in training_sentences:
            posInSent=0
            tk_final_position=len(sentd)-1
            for tkn in sentd:         
                vecTkn=sbd.vectorizeTokenWithPOSContext(tk_position,context_window_size,full_doc)
                if posInSent==tk_final_position:
                    #If the token is a boundary add a positive label "1" to the labels array
                    train_labels.append(1)
                    print(tkn[0]+" <1>")
                    #If the token is not a boundary add a negative label "0" to the label array
                else:
                    train_labels.append(0)
                    print(tkn[0]+" <0>")
                    #position index of the token in the sentence
                posInSent+=1
                tk_position+=1
                print(vecTkn)
                train_matrix.append(vecTkn)
            
    return train_matrix,train_labels

def makeTrainingSet3(training_docs,context_window_size):
    train_matrix=[]
    train_labels=[]
    
    for doc in training_docs:
        training_sentences=brown.tagged_sents(doc,tagset='universal')
        sbd_indx=sbd.getEOSIndicesOfDoc(training_sentences)
        full_doc=brown.tagged_words(doc,tagset='universal')
        tk_position=0
        for tkn in full_doc:         
            vecTkn=sbd.vectorizeTokenWithPOSContext(tk_position,context_window_size,full_doc)
            if tk_position in sbd_indx:
                #If the token is a boundary add a positive label "1" to the labels array
                train_labels.append(1)
                print(tkn[0]+" <1>")
                #If the token is not a boundary add a negative label "0" to the label array
            else:
                train_labels.append(0)
                print(tkn[0]+" <0>")
                #position index of the token in the sentence
            tk_position+=1
            print(vecTkn)
            train_matrix.append(vecTkn)
            
    return train_matrix,train_labels

def makeTrainingSet2(training_sentences,context_window_size):
    train_matrix=[]
    train_labels=[]
    for sent in training_sentences:
        #print(sent)
    #control variables, token index position in a sentence and the last index of the sentence (last token and boundary of the sentence)
        tk_position=0
        tk_final_position=len(sent)-1
        #represent each token in the sentence as a vector of the POS tags of its context(surrounding tokens, k to the left and k to the right)
        for tkn in sent:
            vecTkn=sbd.vectorizeTokenWithPOSContext(tk_position,context_window_size,sent)
            #If the token is the last one in the sentence, it means it is boundary token, therefore a positive example
            if tk_position==tk_final_position:
                #If the token is a boundary add a positive label "1" to the labels array
                train_labels.append(1)
            #If the token is not a boundary add a negative label "0" to the label array
            else:
                train_labels.append(0)
            #position index of the token in the sentence
            tk_position+=1        
            train_matrix.append(vecTkn)
#            print(tkn)
#            print(vecTkn)
#            print(train_labels[tk_position-1])
    return train_matrix,train_labels

#General argument parsing of the program
parser = argparse.ArgumentParser(description='This program vectorizes the all the tokens in a set of sentences from the annotated brown corpus (as included in the nltk). Each vector represents the context of a given token. The context is given by the surrounding tokens. The final output this program produces is a matrix to serve as training set for a machine learining algorithm.')
#index is the starting point from which we will take the training sentences
parser.add_argument('index', metavar='start_index', nargs='?', default=1, type=int, help='The starting index from which we will take the training sentences. Default=0')
#n is the number of documents for the training set
parser.add_argument('n', metavar='num_sents', nargs='?', default=500, type=int, help='It is the number of documents for the training set. Default=500 (all the brown corpus)')
#k is the size of the window for the context por representing the tokens. It represents the number of tokens we will take before and after the evaluated token
parser.add_argument('k', metavar='context_size', nargs='?', default=2, type=int, help='It is the size of the window for the context por representing the tokens. It represents the number of tokens we will take before and after the evaluated token. Default=2')
#Directory to store the output files
parser.add_argument('target', metavar='output_dir', nargs='?', default=".", type=str, help='Directory path to store the output files. Default=.')
args=parser.parse_args()

index=args.index
n=args.n
k=args.k
target_dir=args.target
#formatting target path if necessary
if target_dir[-1]!='/':
    target_dir=target_dir+'/'
#creating target path if necessary
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

#get all the documents in our POS annotated corpus
#sentences = brown.tagged_sents(tagset='universal')
corpus_ids=brown.fileids()
#Get a certain number of documents to get example sentences 
#observed_examples=sentences[index:index+n]
observed_examples=corpus_ids[index:index+n]
#Compute the training set matrix and its labels
trainX,trainY=makeTrainingSet(observed_examples,k)
#serialize the generated training set and labels in target files
training_mat_file=open(target_dir+"training_set_matrix_k"+str(k)+"_n"+str(n)+"_i"+str(index)+"_pickle.bin",'wb')
training_lab_file=open(target_dir+"training_set_labels_k"+str(k)+"_n"+str(n)+"_i"+str(index)+"_pickle.bin",'wb')
pickle.dump(trainX,training_mat_file)
pickle.dump(trainY,training_lab_file)
#print(trainX)
#print(trainY)