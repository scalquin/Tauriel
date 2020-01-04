#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:36:12 2019

@author: sebastian
"""
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import requests
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time
from utils import *
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
#from alerta import enviaralerta
import cv2
from convertirImagen import base64code
from alerta2 import enviaralerta2

def read_image(path):
    """ Method to read an image from file to matrix """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def plot_image(image, title=''):
    """ It plots an image as it is in a single column plot """
    # Plot our image using subplots to specify a size and title
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title(title)
    ax1.imshow(image)
    
def get_faces(image):
    """
    It returns an array with the detected faces in an image
    Every face is defined as OpenCV does: top-left x, top-left y, width and height.
    """
    faces = 0
    # To avoid overwriting
    try:
        image_copy = np.copy(image)
    
    # The filter works with grayscale images
        gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
        face_classifier = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
    
    # Detect the faces in image
        faces = face_classifier.detectMultiScale(gray, 1.2, 5)
        
        return faces
    except:
        return faces

def draw_faces(image, faces=None, plot=True):
    """
    It plots an image with its detected faces. If faces is None, it calculates the faces too
    """
    if faces is None:
        faces = get_faces(image)
    
    # To avoid overwriting
    image_with_faces = np.copy(image)
    
    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_faces, (x,y), (x+w,y+h), (255,0,0), 3)
        
    if plot is True:
        plot_image(image_with_faces)
    else:
        return image_with_faces
    
    

def plot_image_with_keypoints(image, image_info):
    """
    It plots keypoints given in (x,y) format
    """
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    
    for (face, keypoints) in image_info:
        for (x,y) in keypoints:
            ax1.scatter(x, y, marker='o', c='c', s=10)
   

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(image)
    

# Load training set
# load_data es un método definido en el fichero utils para cargar las imágenes.
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))

# Import deep learning resources from Keras


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


# Summarize the model
model.summary()

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, History  

epochs = 150
histo = History()

## Compile the model
def compile_model(model, epochs):
    
    filepath = './ModeloIdentificacion/model.hdf5'
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=filepath, 
                                   verbose=1, save_best_only=True)

    ## Train the model
    hist = model.fit(X_train, y_train, validation_split=0.2,
              epochs=epochs, batch_size=20, callbacks=[checkpointer, histo], verbose=1)
    
    model.save(filepath)
    
    return hist

def show_training_validation_loss(hist, epochs):
    plt.plot(range(epochs), hist.history[
             'val_loss'], 'g-', label='Val Loss')
    plt.plot(range(epochs), hist.history[
             'loss'], 'g--', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 

# TODO: Set True if you want to train the network. It will get a pretrained network values from a file.
train_net = False

if train_net is True:
    hist = compile_model(model, epochs) 
else:
    model.load_weights('./ModeloIdentificacion/model.hdf5')
    

def get_keypoints(image, faces=None):
    
    # list of pairs (face, keypoints)
    result = []
    
    if faces is None:
        faces = get_faces(image)
    
    # Same size than training/validation set
    faces_shape = (96, 96)
    
    # To avoid overwriting
    image_copy = np.copy(image)
    
    # For each face, we detect keypoints and show features
    for (x,y,w,h) in faces:

        # We crop the face region
        face = image_copy[y:y+h,x:x+w]

        # Face converted to grayscale and resize (our CNN receives images of 96x96x1)
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resize_gray_face = cv2.resize(gray_face, faces_shape) / 255

        # Formatting x inputs. Inputs will have format of (1, 96, 96, 1)
        inputs = np.expand_dims(np.expand_dims(resize_gray_face, axis=-1), axis=0)
                                
        # Get keypoints result                        
        predicted_keypoints = model.predict(inputs)

        # All keypoints in a single flat array. We will retrieve keypoints as (x,y) with (idx, idx+1) values.
        predicted_keypoints = np.squeeze(predicted_keypoints)
        
        keypoints = []        
        for idx in range(0, len(predicted_keypoints), 2):
            # Scale factor (revert scale)
            x_scale_factor = face.shape[0]/faces_shape[0] 
            y_scale_factor = face.shape[1]/faces_shape[1] 

            # Offset of the center of the scatter
            x_center_left_offset = predicted_keypoints[idx] * faces_shape[0]/2 + faces_shape[0]/2 
            y_center_left_offset = predicted_keypoints[idx + 1] * faces_shape[1]/2 + faces_shape[1]/2
            
            x_center = int(x + (x_scale_factor * x_center_left_offset))
            y_center = int(y + (y_scale_factor * y_center_left_offset))

            keypoints.append([x_center, y_center])
        
        result.append([(x,y,w,h), keypoints])
    
    return result

def show_image_and_features(image_path):
    image = read_image(image_path)
    faces = get_faces(image)
    keypoints = get_keypoints(image, faces)
    image_with_faces = draw_faces(image, faces ,plot=False)
    plot_image_with_keypoints(image_with_faces, keypoints)
    return True

#show_image_and_features('images/breaking_bad.jpg')import numpy as np


longitud, altura = 224, 224

modelo = "/home/sebastian/Documentos/Tesis3/Tesis/CodigoNeuronaRF/ai-reconocimiento-facial-python-master/TesisModel/modelo.h5"
pesos_modelo = "/home/sebastian/Documentos/Tesis3/Tesis/CodigoNeuronaRF/ai-reconocimiento-facial-python-master/TesisModel/pesos.h5"
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    cnn = load_model(modelo)
    cnn.load_weights(pesos_modelo)

def predict(file, probabilidad):    
    name = ""
    
    identi = cv2.imread(file)
    identi = cv2.resize(identi,(224,224))
    identi = identi.astype('float32')
    identi = np.expand_dims(identi, 0)
    id_ = cnn.predict(identi)
    prob = np.max(id_)
    label = np.argmax(id_)
    
    name = str(label)+" "+str(prob*100)[:5]+"%"
    predic = [label, (prob*100)]
    return predic
# Create instance of video capturer

count = 0
# Try to get the first frame
from alerta import enviaralerta
# keep video stream open
cap = cv2.VideoCapture(0)
#ap = cv2.VideoCapture('http://192.168.0.5:8080/video')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    caras = get_faces(frame)
    caras = format(len(caras))
    if caras == "1":
        print(str(caras))
        cv2.imwrite("ImgTemp/img"+str(count)+".jpg",frame)
        image = read_image("ImgTemp/img"+str(count)+".jpg")
        path_img = ("ImgTemp/img"+str(count)+".jpg")
        p = predict(path_img, 0.6)
        #print(p[0],p[1])
        show_image_and_features(path_img)
        count = count+1
        id = str(p[0])
        #print(p[1])
        #print(id)
        #time.sleep(2.000)
        if p[1] < 90.0:
            print("Enviar Alerta"+path_img)
            ####-----------------------------
            imagen = base64code(path_img)
            #print(imagen)
            enviaralerta(id,imagen)
            print("Ingreso!!")
            ###------------------------------
            #r = requests.post("http://10.100.54.71:8082:/alerts/AddAlert", data={1,1,1,'23-12-2019/11:53'})
        print("Enviar dato a BD:"+str(p[0]))
        enviaralerta2(id)
    #Solo Usar con camara integrada
    #time.sleep(0.50) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()