
import cv2
from Prediccion import Prediccion

clases=["Manzana","Banano","Limón","Mango","Pera"]

ancho=256
alto=256

miModeloCNN=Prediccion("models/modeloFrutasUno.h5",ancho,alto)
imagen=cv2.imread("dataset/test/3/3_ (4).jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()