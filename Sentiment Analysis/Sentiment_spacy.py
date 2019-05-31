#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:11:51 2019

@author: Ruman
"""
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

nlp = English()
tokenizer = Tokenizer(nlp.vocab)



sentences = ["VADER is smart, handsome, and funny.", # ejemplo de frase positiva
"VADER is smart, handsome, and funny!", # Detección del énfasis exclamativo (intensidad de sentimiento incrementada)
"VADER is very smart, handsome, and funny.", # Detección de palabras aumentativas (intensidad  incrementada)
"VADER is VERY SMART, handsome, and FUNNY.", # énfasis derivado del uso de mayúsculas
"VADER is VERY SMART, handsome, and FUNNY!!!",# cominación de las anteriores
"VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# combinación de las anteriores al máximo nivel
"The book was good.", # frase positiva
"The book was kind of good.", # disminución de positividad (ajuste de intensidad)
"The plot was good, but the characters are uncompelling and the dialog is not great.", # negación
"A really bad, horrible book.", # frase negativa con potenciadores de intensidad
"At least it isn't a horrible book.", # negación de negatividad
":) and :D", # emoticones
"", # strings vacíos son tratados correctamente
"Today kinda sux", #detección de palabras de slang
"Today KINDA SUX!", # combinación de slung con mayúsculas y exclamación (incrementa sentimiento)
"I'll get by", # Esta frase es neutra
"Today kinda sux, think I'll get by", # Este ejemplo sirve para comparar con la frase siguiente    
"Today kinda sux, but I'll get by" # 'pero' suaviza la negatividad de la frase anterior
]

for sentence in sentences:
    doc=nlp(sentence)
    print(doc.text)
    for token in doc:
        # Check if the token resembles a number
        print(token.text)
        print(token.sentiment)


#Devuelve todo ceros. Es solo el parámetro para almacenar esta info, pero no lo devuelve
        
#Ver -> https://github.com/explosion/spacy/blob/master/examples/deep_learning_keras.py