#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:40:23 2019

@author: Ruman
"""

from __future__ import print_function
from nltk.chat.util import Chat, reflections

# a table of response pairs, where each pair consists of a
# regular expression, and a list of possible responses,
# with group-macros labelled as %1, %2.

pairs = (
    
    (
        r'(.*) incidencia (.*)',
        (
            "Con que proyecto está relacionada la incidencia?",
            "De que proyecto me está hablando?",
        ),
    ),    
    (
        r'incidencia (.*)',
        (
            "Con que proyecto está relacionada la incidencia?",
            "De que proyecto me está hablando?",
        ),
    ),           
    (
        r'(.*)incidencia',
        (
            "Con que proyecto está relacionada la incidencia?",
            "De que proyecto me está hablando?",
        ),
    ),      
        
    (
        r'(.*)',
        (
            "Lo siento, no comprendo lo que quiere decir. ¿Podría tratar de explicarmelo con otras palabras?",
            "Disculpe, no entiendo como puedo aydarle",
        ),
    ),
)
        
reflections = {
  "yo soy"       : "tu eres",
  "yo era"      : "tu eras",
  "yo"          : "tu",
  "mio"        : "tuyo",
  "yo seré"        : "tu serás",
  "yo tengo"       : "tu tienes",
  "mi"         : "tu",
  "mio"        : "tuyo",
  "tu eres"    : "Yo soy",
  "tu eras"   : "Yo era",
  "tu tienes"     : "Yo tengo",
  "Tu serás"     : "Yo seré",
  "Tuyo"       : "mio",
  "Tus"      : "mis",
  "Tu"        : "yo"
}        



def eliza_chat():
    print("Bienvenido a RLS consuting\n---------")
    print('En que podemos ayudarle? Escriba "quit" para salir.')
    print('=' * 72)
    eliza_chatbot = Chat(pairs, reflections)
    eliza_chatbot.converse()

if __name__ == "__main__":
    eliza_chat()

