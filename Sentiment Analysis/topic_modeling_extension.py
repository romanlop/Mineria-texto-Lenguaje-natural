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
#Importamos el vectorizador tfidf
from sklearn.feature_extraction.text import TfidfVectorizer


categories = 'rec.sport.hockey','sci.space'

print("Cargamos las categorías del dataset 20newsgroups:", categories)
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
print('Número de documentos cargados del dataset 20newsgroups para esas 2 categorias:',"%d documentos" % len(dataset.data))
#print()
labels = dataset.target
print (dataset.target)




#Noticias destacadas ELPAIS
text = ["""1. Anticorrupción se planta ante las dilaciones de la juez de los ERE del PSOE. 
La instrucción de las causas de corrupción que afectan a la Junta de Andalucía se ha convertido en algo parecido a un 
combate de boxeo en el que en lugar de golpes se intercambian escritos cargados de metralla jurídica. 
A un lado de este «cuadrilátero» judicial se sitúa la juez María Núñez Bolaños y buena parte de las defensas, 
que han visto en ella a una aliada. En medio está la acusación que ejerce la Junta, gobernada desde enero por el PP 
y Ciudadanos. En el otro rincón del «ring», los fiscales y la acusación popular del PP andaluz, que cuestionan la 
manera de instruir de la magistrada, su exclusión masiva de políticos en piezas separadas del caso ERE, sus más de 
34 archivos por caducidad de delitos, el retraso de las diligencias, la falta de imputaciones... Es una espiral que v
iene de lejos, pero parece haber cruzado una línea de difícil retorno esta misma semana. La Fiscalía Anticorrupción ha explotado. 
En un escrito del 3 de mayo elevado al juzgado, denuncia la «pasividad evidente» de la juez ante las prescripciones en una de las 
piezas sobre ayudas irregulares en las que se ha parcelado la macrocausa penal de los ERE.""",
"2. El exnovio de Verónica, que quedó ayer en libertad sin cargos, niega ser el autor de la filtración del vídeo. A. T., el exnovio y el que era principal sospechoso de la difusión del vídeo con imágenes íntimas de Verónica Rubio, la mujer empleada de Iveco que se quitó la vida el pasado 25 de mayo, se entregó voluntariamente ayer por la tarde, alrededor de las 16.00 horas, en el puesto de la Guardia Civil de la población donde reside, Mejorada del Campo, al este de Madrid. El hombre se identificó como «la expareja de la chica que había quitado la vida el otro día», confirmaron fuentes del Instituto Armado, que puso al hombre a disposición de la Policía Nacional de Coslada, que investiga el caso. Dos agentes de la Policía se personaron en este cuartel y le trasladaron a las instalaciones de la Brigada Central de Investigación Tecnológica de la Jefatura Superior de Policía de Madrid, donde se le tomó declaración, según informaron fuentes policiales a ABC. Negó haber sido la filtración del vídeo. Al cierre de esta edición, quedó en libertad sin cargos al no hallarse «indicios de criminalidad contra él».",
"3. Los socialistas eligen al nacionalismo en Navarra con el aval de Ferraz. Todavía estaría la candidata socialista y secretaria general del PSN, María Chivite, desayunando con la portada de ABC que se hacía eco del «chantaje» del PNV al Gobierno de Pedro Sánchez, cuando descolgó el teléfono y llamó a la presidenta en funciones del Ejecutivo navarro, Uxue Barkos, para iniciar las negociaciones de cara a la formación de un «gobierno progresista». La misma operación repitió con los líderes de Podemos, Eduardo Santos, y de Izquierda Ezkerra, Marisa de Simón. Todos se mostraron dispuestos a alcanzar un acuerdo que evite que el partido más votado en las elecciones del pasado domingo, Navarra Suma, alcance el gobierno. Y pese a que ni Podemos ni Izquierda Ezkerra se definen como nacionalistas, no tuvieron ningún problema en que participe en la coalición una formación como Geroa Bai, la marca navarra del PNV.",
"4. Pugna israelí por armas al futuro vehículo de combate del Ejército español. Las dos principales compañías de defensa terrestre israelíes, Elbit Systems y Rafael, han desembarcado con fuerza en la primera edición de la Feria Internacional de Defensa y Seguridad (Feindef) que se celebra esta semana en Madrid. ¿El motivo? El jugoso contrato para armar al futuro vehículo blindado de combate 8x8 «Dragón» del Ejército de Tierra, cuyo programa contempla una fase inicial de construcción de 348 vehículos (2.100 millones de euros) de los cuales alrededor de 190 unidades, según fuentes del sector, irán armados con una torre no tripulada de cañón de 30 mm. (valoradas en 350 millones en total). En esa segunda parte es donde Elbit Systems y Rafael se la juegan.",
"5. La sombra del «impeachment» amenaza de nuevo a Trump. A pesar de que la Casa Blanca intenta por todos los medios que el caso de la «trama rusa» quede cerrado de una vez por todas, Donald Trump exhibió este jueves su profundo malestar con el fiscal especial Robert Mueller, quien tras investigarle se ha negado a proclamar sus inocencia, impeliendo a los demócratas a pedir de nuevo su recusación o «impeachment». Sin embargo, quien debe decidir sobre ese polémico juicio en el Capitolio, la presidenta de la Cámara de Representantes, Nancy Pelosi, no está de momento por la labor, calificándola de «quijotesca». Desde primera hora de ayer, Trump aireó su irritación con el cierre en falso de la investigación de Mueller. Antes de las ocho de la mañana, hora de Washington, cometió un desliz en la red social Twitter, publicando un mensaje en el que decía: «No tuve nada que ver con que Rusia me ayudara a ser elegido».",
"6. El Real Madrid prepara un sistema a medida de Hazard. Es el futbolista ideal para crear el fútbol de ataque del Real Madrid. Un director de juego ofensivo. Tiene pase desde la banda y en profundidad por el centro. Y es un mediocampista con gol. Se despide del Chelsea con 21 goles y 17 asistencias en su última temporada, los mejores números de su carrera. El club blanco y Zidane luchan por el fichaje de Hazard desde hace tres años. Falta que los dos clubes concreten el precio definitivo. El nuevo proyecto de Zinedine va a juntar en el once al Balón de Oro y al Balón de Plata del Mundial de Rusia. La estrella está en camino . Ramos será su capitán. Y que el capitán siga también es noticia. El anhelo del belga es firmar por el Real Madrid antes de concentrarse con su selección. Las dos entidades hablaron ayer y el pacto se acerca."]

#El pais
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=5)
#vectorizer = TfidfVectorizer()
tfMatrix = vectorizer.fit_transform(text) #El vectorizador aprende el vocabulario del corpus
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



#Textos Hockey
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
tf_feature_names1 = vectorizer.get_feature_names()

topics = 4
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
lda_model = LatentDirichletAllocation(n_components =topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfMatrix)
H1= lda_model.components_
W1 = lda_model.transform(tfMatrix)



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

display_topics(H1, W1, tf_feature_names1, tfMatrix, no_top_words, no_top_documents)