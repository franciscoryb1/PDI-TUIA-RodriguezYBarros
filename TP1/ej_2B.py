import cv2
import numpy as np
import matplotlib.pyplot as plt

from ej_2_fran import analizar_imagen_rasgos_letras

examenes = ["TP1/multiple_choice_1.png",
            "TP1/multiple_choice_2.png",
            "TP1/multiple_choice_3.png",
            "TP1/multiple_choice_4.png",
            "TP1/multiple_choice_5.png"]

for examen in examenes:
    # Cargo la imagen
    img = cv2.imread(examen,cv2.IMREAD_GRAYSCALE)
    # Corto el encabezado
    encabezado = img[108:130, 30:750]
    # Corto cada campo
    nombre = encabezado[0:20, 70:250]
    id = encabezado[0:20, 300:400]
    code = encabezado[0:20, 460:535]
    fecha = encabezado[0:20, 619:725]
    # Obtengo los datos de cada campo
    d_nombre = analizar_imagen_rasgos_letras(nombre)
    d_id = analizar_imagen_rasgos_letras(id)
    d_code = analizar_imagen_rasgos_letras(code)
    d_fecha = analizar_imagen_rasgos_letras(fecha)


    print("EXAMEN:", examen)
    # Nombre
    if d_nombre["Caracteres"] > 25 or d_nombre["Caracteres"] == 0 or  d_nombre["Palabras"] < 2:
        print("Nombre: Mal")
    else:
        print("Nombre: Ok")
    
    # Id
    if d_id["Caracteres"] > 8 or d_id["Caracteres"] == 0 or d_id["Palabras"] > 1:
        print("Id: Mal")
    else:
        print("Id: Ok")
    
    # Code
    if d_code["Caracteres"] == 1:
        print("Code: Ok")
    else:
        print("Code: Mal")
    
    # Fecha
    if d_fecha["Caracteres"] > 8 or d_fecha["Caracteres"] == 0 or d_fecha["Palabras"] > 1:
        print("fecha: Mal")
    else:
        print("fecha: Ok")



    


    # print("EXAMEN:", examen)
    # #Imprimo d_nombre
    # print("- NOMBRE:")
    # print("--- CARACTERES:", d_nombre["Caracteres"])
    # print("--- PALABRAS:", d_nombre["Palabras"])

    # #Imprimo d_id
    # print("- ID:")
    # print("--- CARACTERES:", d_id["Caracteres"])
    # print("--- PALABRAS:", d_id["Palabras"])

    # #Imprimo d_code
    # print("- CODE:")
    # print("--- CARACTERES:", d_code["Caracteres"])
    # print("--- PALABRAS:", d_code["Palabras"])

    # #Imprimo d_fecha
    # print("- FECHA:")
    # print("--- CARACTERES:", d_fecha["Caracteres"])
    # print("--- PALABRAS:", d_fecha["Palabras"])



