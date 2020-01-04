#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 01:20:02 2019

@author: sebastian
"""


import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications

vgg=applications.vgg16.VGG16()

vgg.summary()

cnn=Sequential()
for capa in vgg.layers:
    cnn.add(capa)
    
cnn.summary()

cnn.pop()

cnn.summary()

for layer in cnn.layers:
    layer.trainable=False

cnn.add(Dense(2,activation='softmax'))

cnn.summary()

def modelo():
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    for capa in vgg.layers:
        cnn.add(capa)
    cnn.layers.pop()
    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(2,activation='softmax'))
    
    return cnn

K.clear_session()

data_entrenamiento = './Gemelos/entrenamiento'

#El mas Optimo hasta el momento
epocas= 10
longitud, altura = 224, 224
batch_size = 10
pasos = 15
validation_steps = 15
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 2
#lr = 0.0004
lr = 0.003

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip= True,
    vertical_flip = True,
    validation_split=0.2,
    rotation_range= 45)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

#validacion_generador = test_datagen.flow_from_directory(
#    data_validacion,
#    target_size=(altura, longitud),
#    batch_size=batch_size,
#    class_mode='categorical')


##CREAR LA RED VGG16

cnn=modelo()

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


cnn.fit_generator(
    entrenamiento_generador, 
    steps_per_epoch = pasos, 
    epochs = epocas)
    #validation_data = validacion_generador, 
    #validation_steps = validation_steps)


target_dir = './NuevoGemelos/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./NuevoGemelos/modelo.h5')
cnn.save_weights('./NuevoGemelos/pesos.h5')
print("Finishim!!")























