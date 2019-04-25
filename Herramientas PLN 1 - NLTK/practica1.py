#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:27:27 2019

@author: Ruman
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import gutenberg

#Vamos a escoger un texto
#print(gutenberg.fileids())

#Cogemos el texto en Raw.
emma = gutenberg.raw('austen-emma.txt')
print("Tamaño enmma:",len(emma))


#*************************************************************************
#2.Dividimos el texto en frases
#*************************************************************************
#sentences = nltk.tokenize.sent_tokenize(emma)
#print ("\n\n2. Frases:",sentences)

#*************************************************************************
#2.Dividimos el texto en frases
#*************************************************************************
sentences = nltk.tokenize.sent_tokenize(emma)
print ("\n\n2. Frases:",sentences[1:6])

#*****************************************************************************
#3.Tokenización: tokenizamos el texto, es decir dividimos el texto en tokens
#*****************************************************************************
tokens = nltk.word_tokenize(emma)
print ("\n\n3. Tokens:",tokens[0:20])