#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:00:26 2019

@author: Ruman
En este capítulo se viene a contar que hay un montón de corpus para trabajar con ellos.
"""

import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown

print(gutenberg.fileids())

#Cogemos el primero de estos textos.
emma = gutenberg.words('austen-emma.txt')
print("Tamaño enmma:",len(emma))

#Para utilizar los comandos vistos en el ejercicio 1 sobre otros textos.
emma = nltk.Text(gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")

#Vamos a sacar algunas estadísticas recorriendo todos los libros de gutemberg
"""for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid)) #OJO, esto cuenta los caracteres blancos como uno mas. 
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    #media de tamaño de las palabras, de las frases, número medio de aparición de palabras.
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid) #round es para redondear a dos decimales
"""

#Acceso al documento en crudo -> raw
print(gutenberg.raw('blake-poems.txt'))
#Dividir el texto en frases. 
black_senteces = gutenberg.sents('blake-poems.txt')
print("Frase número 20 de black sentences:", black_senteces[20])
#Frase mas larga del libro
longest_len = max(len(s) for s in black_senteces)

"""
Dristribuciones de Frecuencia Condicionales
"""
text = ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said']
pairs = [('news', 'The'), ('news', 'Fulton'), ('news', 'County')]
