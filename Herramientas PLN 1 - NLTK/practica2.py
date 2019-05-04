#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:39:29 2019

@author: Ruman
"""
# 1. Importamos el corpus CESS del español, que es una colección de textos anotados
from nltk.corpus import cess_esp #este corpus usa el siguiente sistema de etiquetas para análisis morfológico.
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.hmm import HiddenMarkovModelTagger


# 2. Cargamos todas las frases anotadas del corpus CESS
sents = cess_esp.tagged_sents() #Devuelve cada palabra con su tag correspondiente. Tag morfológico.

# 3. Creamos un conjunto de entrenamiento y otro de prueba
#Metemos en el conjunto de entrenamiento el 90% de las frases, y el restante 10% en el conjunto de test
training = []
test = []
for i in range(len(sents)):
    if i % 10:
        training.append(sents[i])
    else:
        test.append(sents[i])

       
    
# 4. Creamos cuatro tipos distintos de analizadores morfológicos: 
# - Un tagger basado en unigramas: aprende de la estadística de cada palabra encontrada en el corpus CESS
# - Otro basadoen bigramas: aprende de la estadística de una palabra y su palabra anterior
# - Otro basado en trigramas: aprende a taggear una palabra basandose en la estadistica de la palabra y sus 2 anteriores
# - Otro basado en Modelos Ocultos de Markov (en inglés Hidden Markov Models, HMM): es el modelo mas completo

unigram_tagger = UnigramTagger(training)
bigram_tagger = BigramTagger(training, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(training, backoff=unigram_tagger)
hmm_tagger = HiddenMarkovModelTagger.train(training)


# 5. Evaluamos sobre el conjunto de test que no usamos para el entrenamiento, para ver qué porcentaje de acierto hemos conseguido
print ('Acierto con unigramas:',unigram_tagger.evaluate(test)*100)
print ('Acierto con bigramas:',bigram_tagger.evaluate(test)*100)
print ('Acierto con trigramas:',trigram_tagger.evaluate(test)*100)
print ('Acierto con HMMs:',hmm_tagger.evaluate(test)*100)

# 6. Vamos a probar algunos textos descargados de internet -> http://delenguayliteratura.com/Analisis_morfologico_I_ejercicios_resueltos.html
texto = "Sombra tuya he de ser. ¿Por qué no está en la cárcel ese infame. A Martirio, aunque es enamoradiza , se le olvidará esto. Estuvo mucho tiempo detrás de ti y le gustabas . Nada lo ha podido evitar . Veo que todo es una terrible repetición."
tokens = nltk.word_tokenize(texto)
tagged = unigram_tagger.tag(tokens)
print(tagged)


#Vamos a probar como funciona con palabras que no existen en el Corpus.
texto = "Los perros son buenos chuchetes."
tokens = nltk.word_tokenize(texto)
print ("TAGGER UNIGRAMA:",tagged)
print ("________________________")
tagged = bigram_tagger.tag(tokens)
print ("TAGGER BIGRAMA:",tagged)
print ("________________________")
tagged = trigram_tagger.tag(tokens)
print ("TAGGER TRIGRAMA:",tagged)
print ("________________________")
tagged = hmm_tagger.tag(tokens)
print ("TAGGER HMMs:",tagged)   #Este si es capaz de identificarla de forma adecuada.
