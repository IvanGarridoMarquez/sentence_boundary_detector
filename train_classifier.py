# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:37:18 2019

@author: Ivan Garrido Marquez
"""

try:
   import cPickle as pickle
except:
   import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import os
#import numpy as np

#General argument parsing of the program
parser = argparse.ArgumentParser(description='This program generates supervised classifier models from training data')
#Directory to store the output files
parser.add_argument('train_x', metavar='training_matrix', type=str, help='File containing a pickle dumped matrix with the examples to train the classifier')
#Directory to store the output files
parser.add_argument('train_y', metavar='training_labels', type=str, help='File containing a pickle dumped list of lables corresponding to the examples for training the classifier')
#Directory to store the output files
parser.add_argument('target', metavar='output_dir', nargs='?', default=".", type=str, help='Directory path to store the output file.')
args=parser.parse_args()
target_dir=args.target
#formatting target path if necessary
if target_dir[-1]!='/':
    target_dir=target_dir+'/'
#creating target path if necessary
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

#Load of training set data, matrix and labels
#training_mat_file=open("training_set_matrix_k2_n5734_i0_pickle.bin",'rb')
#training_lab_file=open("training_set_labels_k2_n5734_i0_pickle.bin",'rb')
training_mat_file=open(args.train_x,'rb')
training_lab_file=open(args.train_y,'rb')
trainX=pickle.load(training_mat_file)
trainY=pickle.load(training_lab_file)

#Creating of an object tree classifier. The parameters, such as maxdepth or information gain, are the defaults
treeClassifier=DecisionTreeClassifier()
#training the tree classifier with our loaded training set and labels
treeClassifier.fit(trainX,trainY)

#name of the training file
train_infix=args.train_x.split('/')[-1].split('.')[0]
#saving the trained tree classification model
model_Tree=open(target_dir+"tree_classifier_model_SBD_"+train_infix+"_pickle.bin",'wb')
pickle.dump(treeClassifier,model_Tree)

#Creating of an object multilayer perceptron classifier. The parameters were chosen as the defaults
mlpClassifier=MLPClassifier()
#training the tree classifier with our loaded training set and labels
mlpClassifier.fit(trainX,trainY)

#saving the trained tree classification model
model_mlp=open(target_dir+"mlp_classifier_model_SBD_"+train_infix+"_pickle.bin",'wb')
pickle.dump(mlpClassifier,model_mlp)

#print(mlpClassifier.predict(np.array([11,5,6,0,0]).reshape(1, -1)))
#print(mlpClassifier.predict(np.array([7,0,6,1,2]).reshape(1, -1)))