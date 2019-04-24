#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:53:12 2019

@author: Ruman
PROCESSING RAW TEXT
"""

import nltk, re, pprint
from nltk import word_tokenize
from urllib import request

"""
Accessing Text from the Web and from Disk
"""

"""url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')""" #En la página indica en que codificación está el fichero.
print("Acceso a los primeros caracteres:", raw[:700])


#Para procesar el texto es necesario tokenizarlo. Quedarnos con palabras y signos de puntuación.
tokens = word_tokenize(raw)
print(type(tokens))
print("Tamaño del fichero raw:", len(raw))
print("Tamaño del fichero tokenizado:", len(tokens))

#Para usar todas las funcionalidades de los capítulos anteriores basta con:
text = nltk.Text(tokens)
text.collocations() #Palabras que aparecen frecuenmente juntas

#Como collocation aparece el nombre del documento que aparece al principio y al final varias veces "Project Gutemberg" Para "cortar" el texto:
raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")
raw = raw[5338:1157743]
raw.find("PART I")



"""
TRATANDO CON HTML
"""