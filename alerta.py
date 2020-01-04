#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:56:03 2019

@author: sebastian
"""
import time

def enviaralerta(id,img):
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    
    hora = time.strftime("%H:%M:%S")
    fecha = time.strftime("%d/%m/%y")
    
    
    url = "http://127.0.0.1:8082/alerts/AddAlerta" 
    post_fields = {'datos': id+","+"1,1"+","+fecha+"-"+hora+","+img}   
    #post_fields = {'datos': "'"+id+"' 1 1 23-12-2019/11:53"}   
    request = Request(url, urlencode(post_fields).encode())
    json = urlopen(request).read().decode()
    print(json)
    