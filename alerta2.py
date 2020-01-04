#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 01:23:45 2019

@author: sebastian
"""


import time

def enviaralerta2(id):
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    
    hora = time.strftime("%H:%M:%S")
    fecha = time.strftime("%d/%m/%y")
    
    
    url = "http://127.0.0.1:8082/alerts/AddAlerta" 
    post_fields = {'datos': id+","+"1,2"+","+fecha+"-"+hora+","+"No"}   
    #post_fields = {'datos': "'"+id+"' 1 1 23-12-2019/11:53"}   
    request = Request(url, urlencode(post_fields).encode())
    json = urlopen(request).read().decode()
    print(json)
    