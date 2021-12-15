import cv2
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
#Red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Conv2D,MaxPool2D,Reshape,Dense,Flatten

inicio = time.time()

def cargarDatos(path,numeroCategorias,limite,width,height):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0, numeroCategorias - 1):
        for idImagen in range(1, limite[categoria] + 1):
            ruta = path + str(categoria) + "/" + str(categoria) + "_ (" + str(idImagen) + ").jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width, height))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento=np.array(imagenesCargadas)
    valoresEsperados=np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados



width=256
height=256
pixeles=width*height
num_chanels=1
img_shape=(width,height,num_chanels)
categorias=5
cantidadDatosEntrenamiento=[200,200,200,200,200]
cantidadDatosPruebas=[50,50,50,50,50]


imagenesEntrenamiento,probabilidadesEntrenamiento=cargarDatos("dataset/train/",categorias,cantidadDatosEntrenamiento,width,height) #Carga de datos para entrenamiento
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",categorias,cantidadDatosPruebas,width,height) #Carga de datos para pruebas

X = np.concatenate((imagenesEntrenamiento, imagenesPrueba), axis=0)
y = np.concatenate((probabilidadesEntrenamiento, probabilidadesPrueba), axis=0)


model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(img_shape))


#capas convolucionales
model.add(Conv2D(kernel_size=7,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=7,strides=2,filters=32,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))



#Aplanamiento 
model.add(Flatten())
model.add(Dense(256,activation="relu"))

#capa salida
model.add(Dense(categorias,activation="softmax"))

#Traducir de keras a tensorflow
model.compile(optimizer="adam", loss="mse", metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

#CrossValidation
accuracy_fold=[]
loss_fold=[]
recall_fold=[]
precision_fold=[]
f1_fold=[]
kFold=KFold(n_splits=5,shuffle=True)
num_fold=1


for train,test in kFold.split(X,y):
    print("training fold", num_fold)
    model.fit(X[train], y[train], epochs=28,batch_size=200)
    metrics=model.evaluate(X[test],y[test])
    f1_fold.append(((2.0 * metrics[3] * metrics[2]) / (metrics[3] + metrics[2])))
    precision_fold.append(metrics[3])
    recall_fold.append(metrics[2])
    accuracy_fold.append(metrics[1])
    loss_fold.append(metrics[0])
    num_fold+=1

#Prueba del modelo
YPred=model.predict(imagenesPrueba)
yPred=np.argmax(YPred, axis=1)
MatrixConf=confusion_matrix(np.argmax(probabilidadesPrueba,axis=1),yPred)

# Guardar modelo
ruta="models/modeloFrutasDos.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()

print("Informe final: ")
print("1. Matríz de confusión: ",MatrixConf)
print("2. Accuracy",np.mean(accuracy_fold))
print("3. Loss",np.mean(loss_fold))
print("4. Recall",np.mean(recall_fold))
print("5. Precision",np.mean(precision_fold))
print("6. F1 Score",np.mean(f1_fold))
print("7. Los siguientes son el Loss, el Accuracy, recall y precisión por interacción del cross validation")
for i in range(0,len(loss_fold)):
    print("Fold ",(i+1),"- Loss(Error)=",loss_fold[i]," - Accuracy=",accuracy_fold[i]," - Recall=",recall_fold[i]," - Precision=",precision_fold[i]," - F1 Score=",f1_fold[i])

fin = time.time()
print("Tiempo en ejecución: ", (fin-inicio))