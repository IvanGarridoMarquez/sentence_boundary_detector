# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:19:52 2019

@author: Ivan Garrido Marquez
"""
import argparse
import tools_for_SBD as sbdt
#from nltk.corpus import brown
import sentence_boundary_detector as sbd
#import nltk
try:
   import cPickle as pickle
except:
   import pickle

def precision(tp,tn,fp,fn):
    return tp/(tp+fp)
    
def recall(tp,tn,fp,fn):
    return tp/(tp+fn)
    
def f1measure(tp,tn,fp,fn):
    return 2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))

def f1measurePR(prec,rec):
    return 2*((prec*rec)/(prec+rec))

def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fp+fn)

clss_algo="mlp"
accu=[]
precs=[]
recs=[]
f1=[]
#for each index
#for i in range(0,100,50):
i=100
    #select(format name of) the classifier
classifier="classifiers/"+clss_algo+"_classifier_model_SBD_training_set_matrix_k2_n50_i"+str(i)+"_pickle_pickle.bin"
    #create a new sentence boundary detector with the chosen classifier    
class_file=open(classifier,'rb')
sbdf=pickle.load(class_file)
    #evaluate for all the rest vectors
    #for every other vector_file
tp=0
tn=0
fp=0
fn=0

test_i=0

matrixf=open("training_set/training_set_matrix_k2_n50_i"+str(test_i)+"_pickle.bin",'rb')
labelsf=open("training_set/training_set_labels_k2_n50_i"+str(test_i)+"_pickle.bin",'rb')
test_matrix_slice=pickle.load(matrixf)
test_labels_slice=pickle.load(labelsf)
        #for each vector in vector_file
ti=0
for vector in test_matrix_slice:
    #classify the vector
    prediction=sbdf.predict(vector.reshape(1,-1))
    #verify the prection vs the original label
    #count the tp, fp, tn, fn
    if test_labels_slice[ti]==0:
        if test_labels_slice[ti]==prediction[0]:
            tn+=1
        else:
            fn+=1
    if test_labels_slice[ti]==1:
        if test_labels_slice[ti]==prediction[0]:
            tp+=1
        else:
            fp+=1
    ti+=1
       
#compute evaluation metrics
ac=accuracy(tp,tn,fp,fn)
accu.append(ac)
ps=precision(tp,tn,fp,fn)
precs.append(ps)
re=recall(tp,tn,fp,fn)
recs.append(re)
f1z=f1measure(tp,tn,fp,fn)
f1.append(f1z)
print(f1)
#output results
print(accu)
print(precs)
print(recs)
print(f1)