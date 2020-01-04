#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:38:03 2019

@author: sebastian
"""
import base64 

def base64code(img):
    image = open(img,'rb')
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    #image_64_decode = base64.decodestring(image_64_encode)
    image_64_encode = image_64_encode.decode('utf-8')
    #print(image_64_encode)
    return image_64_encode

imagenkeanuuuu='WhatsApp Image 2019-12-24 at 00-Copy1.07.14.jpg'
base64code(imagenkeanuuuu)