#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:11:18 2019

@author: Ruman

Web Scraping -> Extraer resultados página marca.
"""

import requests
from urllib.request import Request, urlopen
import time
from bs4 import BeautifulSoup  #Librería para extraer contenido de ficherso HTML y XML



start_dt = 1
end_dt = 30

for dt in range(1,2):
    url='https://www.siguetuliga.com/liga/galicia-tercera-division-grupo-1/jornada-'+str(dt)
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    """print(webpage)"""
    time.sleep(2)
    soup = BeautifulSoup(webpage, 'html.parser')
    x=soup.prettify()
    print(soup.find(id="idEquipoLocal-1086438"))



