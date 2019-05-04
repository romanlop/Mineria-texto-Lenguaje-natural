#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:54:55 2019

@author: Ruman
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import gutenberg
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import spacy #otro tipo de lematización. Mejor dicho es para procesamiento avanzado de lenguale natural.
from spacy import displacy  

#Cadena de texto
texto = "I didn't notice my animals were uglier than yours! I'm sorry..."

#*************************************************************************
#2.Dividimos el texto en frases
#*************************************************************************
frases = nltk.tokenize.sent_tokenize(texto)
print ("\n\n2. Frases:",frases)


#*****************************************************************************
#3.Tokenización: tokenizamos el texto, es decir dividimos el texto en tokens
#*****************************************************************************
tokens = nltk.word_tokenize(texto)
print ("\n\n3. Tokens:",tokens)


#*****************************************************************************
#4.De paso aprovechamos para poner la etiqueta POS a cada token. Lo hacemos agrupado por frase.
#*****************************************************************************
tags = nltk.pos_tag(tokens)
print ("\n\n4. Tags:",tokens)

#*******************************************************************  
#5.Stemming: obtenemos la raíz (en inglés 'stem') de cada token
#*******************************************************************  
stemmer = PorterStemmer()
raiz=[]
for tok in tokens:
    raiz.append(stemmer.stem(tok.lower()))
print ("\n\n5. Stems: ",raiz)

 
#*******************************************************************  
#6.Lematización: obtenemos el lema de cada token 
#EXISTEN DiFERENTES TIPOS DE LEMATIZADORES
#*******************************************************************  
lemmatizer = WordNetLemmatizer()
lemma=[]

from nltk.corpus import wordnet
wnTags = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV} 
print ("\n\n\n6. Lemas: ")
for (tok,tag) in tags:
    #wordnet no contiene las formas abreviadas 'm  y  n't así que las introducimos nosotros para que lematice bien
    if tok=='\'m':
        tok = 'am'
    if tok=='\'s':
        tok = 'is'
    if tok=='n\'t':
        tok = 'not'
    tag = tag[:1] #Nos qudamos solo con la primera letra del TAG para simplificar.
    lemma.append(lemmatizer.lemmatize(tok.lower(),wnTags.get(tag,wordnet.NOUN)))
    if lemma is None: #Si wordnet no contiene la palabra, supondremos que el lema es igual al token
       lemma = tok.lower() 
print ("Lematización WordNet:",lemma)


#tiene diferentes modelos para diferentes idiomas, entre ellos el castellano. Es necesario instalarlos. https://spacy.io/usage/models
#keeping only tagger component needed for lemmatization
nlp = spacy.load('en_core_web_sm')
#Parseamos el texto.
doc = nlp(texto)
for token in doc:
    #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    print("Resultado:",token.text, token.lemma_, token.pos_, token.tag_)
    
#*******************************************************************    
#Vamos a representar Tokens / TAGS y LEMAS
#******************************************************************* 
print("\n TAGS:",tags)
print("\n LEMAS:",lemma)
#Generamos la gramática para los tags utilizando el array creado para los tags

#*******************************************************************    
#Análisis sintácnico
#******************************************************************* 
nltk.pos_tag(lemma)
grammar_POS = nltk.CFG.fromstring("""
S -> NP VP
NP -> 'NNS' 'VBP' 'RB' | 'RB' 'IN' 'UH' Punt |'NN' VP
VP -> 'VB' 'PRP$' 'JJ' 'VB' NP | 'VB' 'JJ' Punt
Punt -> '.' NP| ':'
""")
#Cargo los tags mediante una sentencia (no he logrado hacerlo en este tiempo con el array que había generado antes).
sentence_POS = "NNS VBP RB VB PRP$ JJ VB RB IN UH . NN VB JJ :".split(" ") 
#Creamos el parser
parser_POS = nltk.ChartParser(grammar_POS)
print ('Analisis sintactico POS:')
#Cargamos en el arbol
for tree_POS in parser_POS.parse(sentence_POS):
    print(tree_POS,'\n')
    tree_POS.draw()

