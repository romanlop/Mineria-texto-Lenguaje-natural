#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:54:16 2019

@author: Ruman
https://scikit-learn.org/0.18/auto_examples/text/document_clustering.html

A continuación, tiene que modificar la práctica anterior para usar un vectorizador tfidf y para 
que coja un corpus mucho más grande que contenga 2 nuevas clases 
(en lugar de las clases 'positivo' y 'negativo', escójase la clase 'atheism' y la clase 'space'). 
El corpus a usar es el de 20newsgroups.
"""

from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


"""
Este corpus tiene las siguientes categorías. Para ello lo cargamos con categories=None y vemos el atributo Target Names del DataSet.
dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=42)
alt.atheism
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
misc.forsale
rec.autos
rec.motorcycles
rec.sport.baseball
rec.sport.hockey
sci.crypt
sci.electronics
sci.med
sci.space
soc.religion.christian
talk.politics.guns
talk.politics.mideast
talk.politics.misc
talk.religion.misc
"""

cat = ['alt.atheism','sci.space']
print("Loading 20 newsgroups dataset for categories:")
dataset = fetch_20newsgroups(subset='all', categories=cat,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
labels = dataset.target


#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(max_df=0.5, max_features=None,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

X = vectorizer.fit_transform(dataset.data)
print("n_samples: %d, n_features: %d" % X.shape)

#Usamos KMEANS para el Clustering
km = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1)
km.fit(X)

#Mostramos los clústers
terms = vectorizer.get_feature_names()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

#Métricas
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))



print("Clasificamos 2 nuevos textos:")
print ()
testCorpus = ["Atheism remains one of the most extreme taboos in Saudi Arabia. It is a red line that no one can cross. Atheists in Saudi Arabia have been suffering from imprisonment, maginalisation, slander, ostracisation and even execution. Atheists are considered terrorists. Efforts for normalisation between those who believe and those who don’t remain bleak in the kingdom. Despite constant warnings of Saudi religious authorities of “the danger of atheism,” many citizens in the kingdom are turning their backs on Islam. The Saudi dehumanizing strict laws in the name of Islam, easy access to information and mass communication are the primary driving forces pushing Saudis to leave religion. Unfortunately, those who explicitly do, find themselves harshly punished or forced to live dual lives."]
testCorpus2 = ["The man speaking was Neil Armstrong, whose brevity marked the moment when the lunar module Eagle completed its perilous journey from Apollo 11 and touched down upon the surface of the Moon. The world waited on tenterhooks as hour after hour of checks were carried out. Finally, the hatch opened, and Armstrong descended the ladder to become the first human to set foot on the Moon, with the now immortal words: That’s one small step for man, one giant leap for mankind.There cannot be many who have not, however briefly, glanced at the Moon and wondered what it must have been like for Armstrong to look back at the blue and green planet we call home. The landing may have happened almost five decades ago, but space exploration has not lost its allure. Even those of us who were not born when this momentous event unfolded are caught in its gravitational pull. With this in mind, it seems only fitting that Sotheby’s New York has decided to host its first space exploration auction, featuring memorabilia from American-led space missions, exactly 48 years to the day after Apollo 11’s lunar landing."]
tfMatrixTest =  vectorizer.transform(testCorpus)
tfMatrixTest2 =  vectorizer.transform(testCorpus2)

print ("TEXTO 1 (sobre el ateísmo en Arabia Saudí):",testCorpus)
print ()
print ("TEXTO 2 (sobre la llegada del hombre a la luna):",testCorpus2)
print ()
print ("Nueva matriz tf1:",tfMatrixTest.toarray() )
print ("Nueva matriz tf2:",tfMatrixTest2.toarray() )
print ()
prediction = km.predict(tfMatrixTest)[0]
print ("Prediccion Texto 1 (sobre el ateísmo): Cluster",prediction)
prediction2 = km.predict(tfMatrixTest2)[0]
print ("Prediccion Texto 2 (sobre la llegada del hombre a la luna): Cluster",prediction2)


