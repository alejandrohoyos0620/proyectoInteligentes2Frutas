from flask import Flask, jsonify, request, make_response
import base64
import cv2
from io import BytesIO
import numpy as np
from PIL import Image
from Prediccion import Prediccion
import os


app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    APIIntroduceParams = {
        'id_client': request.json['id_client'],
        'images': request.json['images'],
        'models': request.json['models']
    }
    results = []
    if decodeBase64(APIIntroduceParams['images']):
        try:
            frutas = generarFrutas('FrutasServidor')
            for model in APIIntroduceParams['models']:
                modeloCNN = initPredict(ancho, alto, model)
                img_id = 0
                resultado = []
                for fruta in frutas:
                    img = cv2.imread("FrutasServidor/"+fruta)
                    index = modeloCNN.predecir(img)
                    resultado.append({'class':clases[index], 'id-image': img_id})
                    img_id+=1
                results.append({'model_id': model, 'results': resultado})
            response = returnSuccessRequests(results)
        except Exception as e:
            response = returnBadRequests()
    else:
        response = returnBadRequests()

    return response


def returnBadRequests():
    response = {
        'state': 'error',
        'message': 'Error making predictions'
    }
    response = jsonify(response)
    resp = make_response(response)
    resp.status_code = 400
    resp.headers['Content-Type'] = 'application/json'
    resp.content_type = 'application/json'
    return resp

def returnSuccessRequests(results):
    response = {
        'state': 'success',
        'message': 'Predictions made satisfactorily',
        'results': results
    }
    response = jsonify(response)
    resp = make_response(response)
    resp.status_code = 200
    resp.headers['Content-Type'] = 'application/json'
    resp.content_type = 'application/json'
    return resp

def decodeBase64(codeImage):
    id = 0
    try:
        for image in codeImage:
            sbuf = BytesIO()
            sbuf.write(base64.b64decode(image['content']))
            pimg = Image.open(sbuf)
            img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
            cv2.imwrite('FrutasServidor/fruta'+str(image['id'])+'.jpg', img)
            id+=1
        return True
    except Exception as e:
        return False

def initPredict(ancho,alto,modelo):
    miModeloCNN = Prediccion("models/" + modelos[modelo], ancho, alto)
    return miModeloCNN

def generarFrutas(ruta):
    dir = ruta
    imagenes = []
    with os.scandir(dir) as ficheros:
        for fichero in ficheros:
            print(fichero.name)
            imagenes.append(fichero.name)
    return imagenes

modelos = {
    '1': "modeloFrutasTres.h5",
    '2': "modeloFrutasUno.h5",
    '3': "modeloFrutasDos.h5",
}
clases=["Manzana","Banano","Limón","Mango","Pera"]
ancho = 256
alto = 256
if __name__ == '__main__':
    app.run(debug=True, port=8069)

