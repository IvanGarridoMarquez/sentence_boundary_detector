# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:53:41 2019

@author: Ivan Garrido Marquez
"""
import numpy as np

#Fuction to map the Universal POS tags to numbers
def postagToNum(tag):
    allPOS=["ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRT","PRON","VERB",".","X"]
    #I added 1 because the 0 would be ambiguous with my representation of the context
    try:
        return allPOS.index(tag)+1
    except:
        return 0

#this function vectorizes a token in the position "index" from the text "txt by taking the k tokens to the right and to the left."
def vectorizeTokenWithPOSContext(index,k,txt):
    tk_position=index
    #Each token tkn is represented as a vector of 2k+3 elements. If there are no tokens at the right or at the left 0 is placed instead
    vecTkn=np.zeros(2*k+3)
    #The first position in the vector is for the POS type of the token
    vecTkn[0]=postagToNum(txt[tk_position][1])
    #Explore the k tokens before and after the token
    for i in range(1,k+1):
        #when there are not any tokens left before the token to vectorize we add 0 to the vector
        if (tk_position-i)<0:
            vecTkn[i]=0
        #otherwise we add to the vector a numerical value associated to the universal POS types
        else:
            vecTkn[i]=postagToNum(txt[tk_position-i][1])
        #we add to the vector the numerical values associated to the tokens after the token to vectorize
        try:
            vecTkn[2*k+1-i]=postagToNum(txt[tk_position+i][1])
        #if we cant explore any further we add 0, this happens for the tokens next to the end of the text
        except:
            vecTkn[2*k+1-i]=0
    if txt[tk_position][0] in ['.','?','!']:
        vecTkn[-2]=1
    try:
        if txt[tk_position+1][0][0].isupper():
            vecTkn[-1]=1
    except:
        vecTkn[-1]=0
    #the fuction returns the produced vector of size 2k+1, filled with integers from 0 to 12
    return vecTkn
    
#This function gets the position indices of every sentence boundary
def getEOSIndicesOfDoc(sents):
    indices=[]
    cursor=0
    inc=0
    for s in sents:
        if inc>0:
            cursor+=len(s)
        else:
            cursor+=len(s)-1#+inc
        indices.append(cursor)
        inc+=1
    return indices

#It returns the positions delimitating a sentence based on the indices caomputed by the previous function
def nextIndices(st,nd,current_end,indx):
    new_st=nd
    new_nd=indx[current_end+1]
    next_ind=current_end+1
    return new_st,new_nd,next_ind