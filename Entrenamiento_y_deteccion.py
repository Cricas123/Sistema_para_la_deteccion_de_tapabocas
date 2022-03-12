import cv2
import os
import numpy as np

direccion = 'C:/Users/Cristian/Documents/INGENIERIA ELECTRONICA/SEMESTRE 6/MICROCONTROLADORES/Proyecto_final_microcontroladores/Imagenes_modelo'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
con = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir

    for fileName in os.listdir(nombre):
        etiquetas.append(con)
        rostros.append(cv2.imread(nombre + '/' + fileName,0))

    con = con + 1

#CREACION DEL MODELO
reconocimiento = cv2.face.LBPHFaceRecognizer_create()
reconocimiento.train(rostros, np.array(etiquetas))
reconocimiento.write('ModeloLBP.xml')
print('MODELO CREADO')
