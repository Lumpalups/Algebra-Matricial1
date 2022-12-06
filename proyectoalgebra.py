#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 03:10:02 2022

@author: mariasalomon
"""

#Cargamos las respectivas librerías que nos permitiran leer y procesar la imagen
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import cmath as cm
import scipy.fftpack as sfft


#Visualización imagen
imagen = io.imread('birthday.jpg')
io.imshow(imagen)
io.show()
io.imsave('birthday.jpg', imagen)

#Notamos que es una imagen rectangular de 757x1600 pixeles

#Para facilitar el tratamiento de la misma convertiremos nuestra imagen RGB a escala de grises:
    
#Lectura como matriz, conversión a escala de grises y cambio de tamaño a matriz cuadrada

imageGray = cv2.imread('birthday.jpg',0)
print(imageGray) #visualización como arreglo  y en escala de grises
imag=cv2.resize(imageGray, (757,757), interpolation=cv2.INTER_CUBIC) #Convertimos a una imagen cuadrada
#Note que el reescalamiento que se hace para cambiar el tamaño consiste en realizar un transformación de coordenadas
#Cuando se recorta la imagen se define una nueva imagen que se rellena con 0 y 1 respectivamente


io.imshow(imag)
io.show()

# 1. SUMA de imágenes como matrices

suma=cv2.add(imag,imag)
print('La suma de imágenes como matrices:')
print (cv2.add(imag,imag))

io.imshow(suma)
io.show()
#Podemos notar como al sumar las dos imágenes la imagen resultante se 'aclara' debido a que los pixeles resultantes son la suma de 0 y 1


#2. RESTA de imágenes como matrices

rest=cv2.subtract(imag,imag)
print('La resta de imágenes como matrices:')
print (cv2.subtract(imag,imag))

io.imshow(rest)
io.show()

#La imagen en negro tiene sentido dado que estamos restando dos imágenes iguales. Si por el contrario definieramos una nueva imagen de la forma:

imag1=2*imag
io.imshow(imag1)
io.show()

rest1=cv2.subtract(imag,imag1)
print('La resta de imágenes como matrices:')
print (cv2.subtract(imag,imag1))

io.imshow(rest1)
io.show()

#3. Multiplicación de matrices como imágenes
 #Multiplicación de la misma imagen
mult=cv2.multiply(imag,imag)
io.imshow(mult)
io.show()
 #Multiplicación de imágenes ligeramente diferentes
 
mult1=cv2.multiply(imag,imag1)
io.imshow(mult1)
io.show()

#Observamos los diferentes patrones que se forman en la imagen haciendo pequeñas variaciones

#Ahora veamos que sucede si se realiza la multiplicación de la matriz(imagen) por un escalar

mult2=cv2.multiply(imag,5) #escalar=5
io.imshow(mult2)
io.show()

#En particular, la multiplicación de matrices es de vital importancia dado que se pueden definir matrices cuyos
#elementos sean todos 0 o 1 que denotamos como 'máscaras' o filtros del mismo tamaño que el de la matriz a la que se la aplicarán
#Este 'mask' se opera sobre la imagen inicial haciendo la respectiva multiplicación elemento a elemento. Su importancia radica sobretodo en
#procesamiento de señales en donde a través de estas máscaras podemos filtras señales o frecuencias que perturban nuestra información. En particular
#cuando de imágenes espaciales hablamos, podemos filtrar ruido y aberraciones o también ajustar los niveles de definición de los pixeles que conforman la imagen
#permitiendo mayor definición.


#Transformaciones

ancho=imag.shape[1] #columnas
alto=imag.shape[0] #filas

#Traslación

M=np.float32([[1,0,100],[0,1,150]]) #Definimos las filas y columnas en donde aplicaremos la transformación
imageOut=cv2.warpAffine(imag,M,(ancho,alto))
io.imshow(imageOut)
io.show()

#Rotación

M1 = cv2.getRotationMatrix2D((ancho//2,alto//2),60,1) #Especificamos el centro de la imagen, ángulo de rotación
imageOut1 = cv2.warpAffine(imag,M1,(ancho,alto))
io.imshow(imageOut1)
io.show()

#Filtraje espacial

#Definimos un filtro circular que deja pasar únicamente las 'frecuencias' espaciales del centro, esto significa que
#los elementos que se encuentren en las orillas no se reflejarán en nuestra nueva imagen:

    
#Butterworth low pass filter
N,N = imag.shape
H= np.zeros((N,N), dtype=np.float32)
D0=40 #corte de frecuencia
n=1 #orden
for u in range(N):
    for v in range(N):
         D=np.sqrt((u-N/2)**2 + (v-N/2)**2)
         H[u,v]=1 / (1+ (D/D0)**(2*n))

plt.figure(5)  
plt.figure(figsize=(10,8))  
plt.title('Cómo se ve el filtro ')
plt.imshow(H, cmap='gray')
plt.show()

#Filtro de frecuencia en el dominio de la imagen
Gshift= imag * H
plt.figure(1)  
plt.figure(figsize=(10,8))  
plt.title("Imagen filtrada al centro ")
plt.imshow(np.abs(Gshift), cmap='inferno')
plt.show()







