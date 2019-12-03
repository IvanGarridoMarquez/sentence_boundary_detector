# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:10:22 2019

@author: Ivan Garrido Marquez
"""

import argparse
import tools_for_SBD as sbdt
from nltk.corpus import brown
import sentence_boundary_detector as sbd
import nltk

#get the input text file
#General argument parsing of the program
parser = argparse.ArgumentParser(description='This program splits an input text file into a list of sentences.')
parser.add_argument('input', metavar='input_text', type=str, help='Input text file to be split in sentences')
parser.add_argument('clasf', metavar='classifier_sbd', nargs='?', type=str, help='Chosen sbd classifier')
parser.add_argument('output', metavar='output_text', nargs='?', type=str, help='Output text file with the sentences')
args=parser.parse_args()

classifier="classifiers/tree_classifier_model_SBD_training_set_matrix_k2_n50_i0_pickle_pickle.bin"
if args.clasf is not None:
    classifier=args.clasf
#create a sentence boundary detector with the chosed trained classifier
sbd_obj=sbd.sentenceBoundaryDetector(classifier,2)
#sbd_obj=sbd.sentenceBoundaryDetector("classifiers/mlp_classifier_model_SBD_training_set_matrix_k2_n50_i0_pickle_pickle.bin",2)

#open the input text file 
ft=open(args.input,'r')
text=ft.read()
#use sentence boundary detector to get all the sentences
sentences,positionsInDoc=sbd_obj.detectSentencesBoundaries(text)

#if an output file argument was given save the sentences there
if args.output is not None:
    fto=open(args.output,'w')
    fto.writelines(sentences)
    
#otherwise send the sentences to standard output
else:
    for sent in sentences:
        print(sent)