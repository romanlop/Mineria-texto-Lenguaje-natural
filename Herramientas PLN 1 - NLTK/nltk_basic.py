# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
from nltk.corpus import treebank

#Tokenización
sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print(tokens)


#Análisis Morfológico
tagged = nltk.pos_tag(tokens)
print(tagged) #POS TAGS -> https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b


#t = treebank.parsed_sents('wsj_0001.mrg')[0]
#t.draw()

entities = nltk.chunk.ne_chunk(tagged)
print(entities)