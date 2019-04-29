#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:27:27 2019

@author: Ruman
Ejemplo del master pero tuneado. Ver fichero Jupyter Practica1
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import gutenberg
from nltk.stem import PorterStemmer

#Vamos a escoger un texto
#print(gutenberg.fileids())

#Cogemos el texto en Raw.
emma = gutenberg.raw('austen-emma.txt')
emma = emma[0:1000]
print("Tamaño enmma:",len(emma))


#*************************************************************************
#2.Dividimos el texto en frases
#*************************************************************************
sentences = nltk.tokenize.sent_tokenize(emma)
print ("\n\n2. Frases:",sentences[0:6])
#Vemos que aparecen muchos caracteres de formato. Vamos a tratar de quitarlos para quedarnos solo con el texto.


#*****************************************************************************
#3.Tokenización: tokenizamos el texto, es decir dividimos el texto en tokens
#*****************************************************************************
tokens = nltk.word_tokenize(emma)
print ("\n\n3. Tokens:",tokens[0:50])


#*****************************************************************************
#4.División en Frases pero con texto tokenizado, por tanto sin saltos de línea por ejemplo.
#5.De paso aprovechamos para poner la etiqueta POS a cada token. Lo hacemos agrupado por frase.
#*****************************************************************************
tokenized_text=[]
tagged=[]
i=0
for s in sentences:
    tokenized_text.append(nltk.word_tokenize(s))  
    tagged.append(nltk.pos_tag(tokenized_text[i]))
    i=i+1  
    
print ("\n\n2. Frases:",tokenized_text[0:6])
print ("\n\n2. Análisis Morfológico:",tagged[0:15])
    
#*******************************************************************  
#6.Stemming: obtenemos la raíz (en inglés 'stem') de cada token
#*******************************************************************  
stemmer = PorterStemmer()
stems=[]
print ("\n\n5. Stems: ")
for tok in tokens:
    stems.append(stemmer.stem(tok.lower()))

print ("Tokens:",tokens[0:50])
print ("\nStems:",stems[0:50])


#*******************************************************************  
#6.Lematización: obtenemos el lema de cada token 
#*******************************************************************  
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
#El lematizador de wordnet solo reconoce 4 etiquetas POS: a (adjetivo), r(adverbio),n (nombre),v(verbo). 
#Así que debemos hacer una conversión del formato Penn Tree Bank al formato wordnet (ej: NN->n, JJ->a, RB->r, VB->V, ...)
from nltk.corpus import wordnet
wnTags = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV} 
print ("\n\n\n6. Lemas: ")
for (tok,tag) in tagged[1]: # lo hacemos para una de las frases taggeadas
    #wordnet no contiene las formas abreviadas 'm  y  n't así que las introducimos nosotros para que lematice bien
    if tok=='\'m':
        tok = 'am'
    if tok=='\'s':
        tok = 'is'
    if tok=='n\'t':
        tok = 'not'
    tag = tag[:1]
    lemma = lemmatizer.lemmatize(tok.lower(),wnTags.get(tag,wordnet.NOUN)) #wordnet.NOUN se pone para que devuelva algo por defecto
    #otra forma alternativa de obtener el lema hubiera sido llamar directamente a la funcion wordnet.morphy, que hace lo mismo:
    #lemma = wordnet.morphy(tok.lower(),wnTags.get(tag,wordnet.NOUN))
    if lemma is None: #Si wordnet no contiene la palabra, supondremos que el lema es igual al token
       lemma = tok.lower() 
    print (lemma)

#*******************************************************************    
#7.Análisis sintáctico
#******************************************************************* 

#Partimos de una frase de un conocido texto de Groucho Marx, con una clara ambigüedad: 
#"While hunting in Africa, I shot an elephant in my pijamas. How he got into my pijamas, I don't know."
#¿Groucho estaba en pijama o el elefante estaba dentro de su pijama?
sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pijamas']

#Creamos nuestra propia Gramatica Libre de Contexto (en inglés CFG)
grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pijamas'
V -> 'shot' | 'did'
P -> 'in'
""")


#Generamos un parser sintáctico capaz de reconocer la gramática
parser = nltk.ChartParser(grammar, trace=1)
print ('\n\n\n7. Analisis sintactico:\n')
for tree in parser.parse(sent):
    print(tree,'\n')
    tree.draw()
nltk.parse.chart.demo(2, print_times=False, trace=1, sent='I saw a dog', numparses=1)