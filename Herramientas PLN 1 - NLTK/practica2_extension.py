#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:18:02 2019

@author: Ruman
Para esta práctica es importante leer el apartado 5 del libro de NLTK -> http://www.nltk.org/book/ch05.html
"""

import nltk
from nltk.corpus import cess_cat 
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from nltk.tag.hmm import HiddenMarkovModelTagger


sents = cess_cat.tagged_sents() 

"""
training = []
test = []
for i in range(len(sents)):
    if i % 10:
        training.append(sents[i])
    else:
        test.append(sents[i])

#Aunque vamos a tratar de centranos en trigram los entrenamos todos.
unigram_tagger = UnigramTagger(training)
bigram_tagger = BigramTagger(training, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(training, backoff=unigram_tagger)
"""

#Evaluamos sobre el conjunto de test que no usamos para el entrenamiento, para ver qué porcentaje de acierto hemos conseguido
print ('Acierto con unigramas:',unigram_tagger.evaluate(test)*100)
print ('Acierto con bigramas:',bigram_tagger.evaluate(test)*100)
print ('Acierto con trigramas:',trigram_tagger.evaluate(test)*100)


#Vamos a trabajar con una frase en catalufo:
frase_cat = "el president de la Generalitat ha tingut 4 chuchetes"
tokens = nltk.word_tokenize(frase_cat)
tagged = trigram_tagger.tag(tokens)
print ("TAGGER UNIGRAMA CAT:",tagged)

#Esta es la forma de aplicar varios taggeadores -> Backoff
#Trabajamos con los datos de entrenamiento 'training_cat' que suponen el 90% del corpus
default_tagger = DefaultTagger ('NLTK_FASHION')
unigram_tagger = UnigramTagger(training, backoff=default_tagger)
bigram_tagger = BigramTagger(training, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(training, backoff=bigram_tagger)
#Entrenamos este modelo
print ('Acierto con unigramas:',unigram_tagger.evaluate(test)*100)
print ('Acierto con bigramas:',bigram_tagger.evaluate(test)*100)
print ('Acierto con trigramas:',trigram_tagger.evaluate(test)*100)

#Lo ejecutamos sobre nuestro palabro
tagged = trigram_tagger.tag(tokens)
print ("TAGGER UNIGRAMA CAT:",tagged)