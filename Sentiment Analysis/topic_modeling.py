#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:38:12 2019

@author: Ruman
"""

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from sklearn.datasets import fetch_20newsgroups


categories = 'alt.atheism','sci.space'

print("Cargamos las categorías del dataset 20newsgroups:", categories)
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
print('Número de documentos cargados del dataset 20newsgroups para esas 2 categorias:',"%d documentos" % len(dataset.data))
#print()
labels = dataset.target
print (dataset.target)


#Importamos el vectorizador tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=5)
#vectorizer = TfidfVectorizer()
tfMatrix = vectorizer.fit_transform(dataset.data) #El vectorizador aprende el vocabulario del corpus
#print("tiempo empleado: %fs" % (time() - t0), ", numero de ejemplos: %d, numero de campos: %d" % X.shape)
#Transformamos los documentos en una matriz de tf's de documentos que es nuestro dataset, donde:
# - cada fila representa una muestra (un documento del corpus)
# - cada columna representa un atributo (la frecuencia de una palabra en dicho documento)
print()
print("Vectorizamos los textos y mostramos la Matriz tf:")
print (tfMatrix.toarray())
print()

print("Mostramos el tamaño del análisis...", "nº de muestras: %d, nº de atributos: %d" % tfMatrix.shape)
tf_feature_names = vectorizer.get_feature_names()


topics = 4
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
lda_model = LatentDirichletAllocation(n_components =topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfMatrix)
H = lda_model.components_
W = lda_model.transform(tfMatrix)

no_top_words = 4
no_top_documents = 3

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("\nTopic %d:" % (topic_idx))
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            print(" ",feature_names[i])
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])

display_topics(H, W, tf_feature_names, tfMatrix, no_top_words, no_top_documents)