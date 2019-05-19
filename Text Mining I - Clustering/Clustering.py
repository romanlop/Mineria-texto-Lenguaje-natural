#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:33:03 2019

@author: Ruman
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

#***************************************************************************************
# 1. Cargamos el corpus de textos
#***************************************************************************************

print("\n\n1. Cargamos el corpus")
trainCorpus = ["Me gustan las vacas",
               "Me gustan los caballos",
               "odio los perros",
               "odio los caballos",
               "me gustan las ranas",
               "me gusta el helado",
               "no quiero comer",
               "Los helados, son cremosos"]
print (trainCorpus)


#*********************************************************************************************************
# 2. Vectorizamos los textos del corpus (convertimos cada texto en un vector de frecuencias de palabras)
#*********************************************************************************************************
print("\n\n2. Vectorizamos los textos")

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(trainCorpus)
#Nos devuelve el conjunto de palabras del corpus. Se puede adicionalmente pasar un diccionario. Ver la docu del comentairo anterior.
print(vectorizer.get_feature_names())
#Tabla con las frecuencias. Las columnas son los Token. Las filas los corpus.
print(X.toarray()) 

#La matriz tf es nuestro dataset, donde:
# - cada fila representa una muestra (un documento del corpus)
# - cada columna representa un atributo (la frecuencia de una palabra en dicho documento)
print("nº de muestras: %d, nº de atributos: %d" % X.shape)


#*********************************************************************************************************
# 3. Clusterizamos los documentos mediante el algoritmo K-means
#*********************************************************************************************************
print("\n\n3. Clusterizamos los textos")

#Asignamos a kmeans un valor de k=2, es decir que el algoritmo intentará encontrar 2 clusters
k=2
km = KMeans(n_clusters=k,max_iter=100)
km.fit(X)
print ("Clusters:",km.labels_)
for i in range(k):
    print ("\nCluster",i,":")
    for j in range(km.labels_.size):
        if km.labels_[j]==i:
            print ("\t",trainCorpus[j])
            
#***************************************************************************************
# 4. Medimos la calidad de nuestro cluster
#***************************************************************************************
print("\n\n4. Medimos la calidad de nuestro cluster")

#Ground truth son las categorias correctas puestas a mano (1= texto positivo, 0=negativo), para comparar con las automaticas de kmeans
groundTruth = [0,0,1,1,0,0,1,0] 
print ("Clusters:    ",km.labels_)
print ("Ground Truth:",groundTruth)


#Un cluster es homogéneo si todos sus elementos contienen miembros de una misma clase
print("Homogeneity: %0.3f" % metrics.homogeneity_score(groundTruth, km.labels_))

#Una clase es completa si todos sus elementos pertenecen al mismo cluster
print("Completeness: %0.3f" % metrics.completeness_score(groundTruth, km.labels_))

print("V-measure: %0.3f" % metrics.v_measure_score(groundTruth, km.labels_))


# 5. Usamos los clusters previos para clasificar un nuevo texto entrante
#***************************************************************************************
print("\n\n4. Clasificamos un nuevo texto entrante")

testCorpus = ["odio los animales"]
tfMatrixTest =  vectorizer.transform(testCorpus)
print ("Nuevo texto:",testCorpus)
print ("Nueva matriz tf:",tfMatrixTest.toarray() )

prediction = km.predict(tfMatrixTest)[0]
print ("Prediccion: Cluster",prediction)