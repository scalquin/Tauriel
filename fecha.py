#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 03:17:43 2019

@author: sebastian
"""

import time

def obtenerfechayhora():
    hora = time.strftime("%H:%M:%S")
    fecha = time.strftime("%d/%m/%y")
    print(hora)
    print(fecha)
    print(fecha+"-"+hora)
obtenerfechayhora()