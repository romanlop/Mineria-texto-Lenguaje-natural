#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:00:48 2019

@author: Ruman
https://www.nltk.org/book/ch01.html
"""

import nltk
from nltk.book import *

#Nos permite obtener el Titulo del texto cargado
print(text2)

#Búsquedas de un termino sobre uno de los textos cargados.
text1.concordance("monstrous")


"""
La palabra mostrous aparece en una serie de contextos determinados, podemos buscar palabras que aparecen en contextos similares
"""
print("\nContexto similar a Mosntuoso:\n")
text1.similar("monstrous")
text2.similar("monstrous")
#Vemos que los dos autores utilizan la palabra con connotaciones difernetes.

"""
Podemos buscar contextos compartidos por varias palabras
"""
print("\nContexto compartido:\n")
text2.common_contexts(["monstrous", "very"])


"""
Podemos ver el número de apariciones y posición en el texto. You can also plot the frequency of word usage through time using https://books.google.com/ngrams
"""
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


"""
Contando Vocavulario
"""
#Tamaño del texto en Tokens, entendido como una agrupación de caracteres. Por ejemplo Hola, :), o por ejemplo una frase que queramos tratar como un Token.
print("Tamaño texto 3:",len(text3))
print("Tamaño texto 2:",len(text2))

#Palabras diferentes en un texto.
#print("Palabras diferentes utilizadas en texto2:",sorted(set(text2)))
print("Número de palabras diferentes utilizzadas en Texto3:", len(set(text3)))

#ocurrencias concretas de una palabra.
print("Ocurrencias de la palabra Smote:", text3.count("smote"))


"""
Estadísticas sencillas
"""
#Distribución de frecuencia de palabras
fdist1 = FreqDist(text3)
print("Distribución de frecuencia en Texto1:", fdist1)
print("50 mas frecuentes:",fdist1.most_common(50))
print("La palabra god aparece:", fdist1['God'])
fdist1.plot(50, cumulative=True)

#Palabras que solo aparecen una vez -> "hapaxes"
print("Palabras que solo aparecen una vez:",fdist1.hapaxes())

"""
Selección de palabras de grano-fino
"""
#Vamos a buscar palabras que tengan mas de 15 caracteres
V = set(text1)
long_words = [w for w in V if len(w) > 15]
print("\nPalabras con mas de 15 caracteres:",sorted(long_words))

#Palabras/Tokens que aparezcan mas de X veces
V = set(text1)
number_words = [w for w in V if fdist1[w] > 25]
print("\nPalabras que aparecen mas de 25 veces:",sorted(number_words))

#Palabras que aparecen juntas a menudo
print("Palabras que aparecen juntas a menudo:")
text4.collocations()


