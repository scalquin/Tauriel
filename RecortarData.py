#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 01:36:19 2019

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, History  
import os
import cv2 

# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)
shape = (96,96)
model = Sequential()
model.add(Convolution2D(16,(2,2),padding='same',input_shape=(96,96, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=3))

model.add(Convolution2D(32,(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

model.add(Convolution2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

model.add(Convolution2D(128,(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(30))
print("Primer Sumario")
model.summary()
model.load_weights('model.hdf5')
print("Segundo Sumario")
model.summary()


def read_image(path):
    """ Method to read an image from file to matrix """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_faces(image):
    """
    It returns an array with the detected faces in an image
    Every face is defined as OpenCV does: top-left x, top-left y, width and height.
    """
    # To avoid overwriting
    image_copy = np.copy(image)
    
    # The filter works with grayscale images
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
    face_classifier = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
    
    # Detect the faces in image
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    
    return faces 

#entrenamiento = './datos-Gemelos/entrenamiento'
#directorio = './datos-Gemelos/entrenamiento1'


def cut_image_entrenamiento(directorio, entrenamiento):
    if not os.path.exists(directorio):
        os.mkdir(directorio)
    lista_de_carpetas = os.listdir(entrenamiento)    
    for car in lista_de_carpetas:
        dire = entrenamiento+'/'+car
        lista_de_archivos = os.listdir(dire)
        for ar in lista_de_archivos:
            image = read_image(entrenamiento+'/'+car+'/'+ar)
            faces = get_faces(image)
            for (x,y,w,h) in faces:
                roi_color = image[y:y+h, x:x+w]
                if not os.path.exists(directorio):
                    os.mkdir(directorio)
                if not os.path.exists(directorio+'/'+car):
                    os.mkdir(directorio+'/'+car)
                img_item = str(directorio+'/'+car+'/'+ar)
                cv2.imwrite(img_item, roi_color)



