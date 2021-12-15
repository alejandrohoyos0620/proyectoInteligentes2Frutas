import cv2
import time

import numpy as np
import base64
import os
import requests
import json

nameWindow="Trackbar del aplicativo"

def crearImagenFrutas(image, contours, num):
    new_img = image
    num_id = num
    for c in contours:
        area = cv2.contourArea(c)
        if area == 0:
            break
        x, y, w, h = cv2.boundingRect(c)
        if w > 320 and h > 110:
            new_img = image[y:y + h, x:x + w]
            print("Presionar c")
            if cv2.waitKey(5) & 0xFF == ord('c'):
                cv2.imwrite('RecortesFront/' + 'fruta_'+ str(num_id) +'.jpg', new_img)
                num_id += 1
    return num_id, new_img

def sendRequest(urlApi):
    data = {'id_client': idcliente, 'images': imgBase64, 'models': modelos}
    headers = {'Content-Type':'application/json'}
    response = requests.post(urlApi, data=json.dumps(data), headers = headers)
    respuesta = json.loads(response.text)
    print(respuesta)

def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,100,700,nothing)
    cv2.createTrackbar("max", nameWindow, 185, 1000, nothing)
    cv2.createTrackbar("kernel", nameWindow, 4, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 120000, 200000, nothing)
    cv2.createTrackbar("areaMax", nameWindow, 180000, 300000, nothing)
    cv2.moveWindow(nameWindow, 600, 500)

def calcularAreas(figuras):
    areas=[]
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

def detectarFormas(imagen,img_id):
    fruta = imagen
    imagengris = cv2.cvtColor(imagen,cv2.COLOR_RGB2GRAY)
    min=cv2.getTrackbarPos("min", nameWindow)
    max=cv2.getTrackbarPos("max", nameWindow)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    areamin = cv2.getTrackbarPos("areaMin", nameWindow)
    areamax = cv2.getTrackbarPos("areaMax", nameWindow)
    bordes = cv2.Canny(imagengris, min, max)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes", bordes)
    cv2.moveWindow("Bordes",850, 0)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    contornosFiguras, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(contornosFiguras)
    i = 0
    for figuraActual in contornosFiguras:
        area = cv2.contourArea(figuraActual)
        if area >= areamin and area <= areamax:
            img_id,imagen = crearImagenFrutas(fruta, contornosFiguras, img_id)
        i+=1
    return fruta,img_id

def cnvrtBase64(rutaImagen):
    img = cv2.imread(rutaImagen)
    retval, buffer = cv2.imencode('.jpg',img)
    jpg_as_text = base64.b64encode(buffer)
    encoded_string = jpg_as_text.decode('utf-8')
    return encoded_string

def generarFrutas(ruta):
    dir = ruta
    imagenes = []
    with os.scandir(dir) as ficheros:
        for fichero in ficheros:
            imagenes.append(fichero.name)
    return imagenes

camara = cv2.VideoCapture(2)

img_id = 0
k = 0
idcliente = input('Por favor ingrese el id del cliente: ')
cantModelos = input('Ahora digite la cantidad de modelos que desea usar: ')
modelos = []

if int(cantModelos) > 0 and int(cantModelos) < 4:
    for i in range(int(cantModelos)):
        modelos.append(input('Para el modelo número '+str(i+1)+ ' seleccione uno de los modelos posibles (1,2,3) '))
        if modelos[i] != "1" and modelos[i] != "2" and modelos[i] != "3":
            print('Este modelo no es correcto')
            exit()
else:
    print('No se pueden generar esa cantidad de modelos.')
    exit()

constructorVentana()
while True:
    k = cv2.waitKey(5) & 0xFF
    _,imagen = camara.read()
    imagen,img_id = detectarFormas(imagen,img_id)
    cv2.imshow("Capturador de pantalla", imagen)
    cv2.moveWindow("Capturador de pantalla", 0, 0)
    if k == ord('e'):
        break

imgBase64 = []
frutas = generarFrutas('RecortesFront')
img_ids = 0
for imagenFruta in frutas:
    imageBase64 = cnvrtBase64('RecortesFront/' + imagenFruta)
    imgBase64.append({"id":img_ids,"content":imageBase64})
    img_ids+=1


apiURL = 'http://localhost:8069/predict'
sendRequest(apiURL)