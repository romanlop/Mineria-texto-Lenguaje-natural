#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:11:18 2019

@author: Ruman

Web Scraping -> Hola Mundo
https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460
"""

import requests
import urllib.request
import time
from bs4 import BeautifulSoup  #Librería para extraer contenido de ficherso HTML y XML

url = 'http://web.mta.info/developers/turnstile.html'
response = requests.get(url)

#Parseamos el HTML con bs4
soup = BeautifulSoup(response.text, "html.parser")
a = soup.findAll('a')

#Hay 36 links antes que este. Por tanto vamos a empezar en el 36.
for i in range(36,len(soup.findAll('a'))+1): #'a' tags are for links
    one_a_tag = soup.findAll('a')[i]
    link = one_a_tag['href']
    download_url = 'http://web.mta.info/developers/'+ link
    urllib.request.urlretrieve(download_url,'./'+link[link.find('/turnstile_')+1:]) 
    time.sleep(1) #pause the code for a sec