import cv2
import numpy as np
import matplotlib.pyplot as plt

from ej_2_fran import analizar_imagen_rasgos_letras

examenes = ["TP1/multiple_choice_1.png",
            "TP1/multiple_choice_2.png",
            "TP1/multiple_choice_3.png",
            "TP1/multiple_choice_4.png",
            "TP1/multiple_choice_5.png"]

nombres = []

for examen in examenes:
    # Cargo la imagen
    img = cv2.imread(examen,cv2.IMREAD_GRAYSCALE)
    # Corto el encabezado
    encabezado = img[108:130, 30:750]
    # Corto cada campo
    nombre = encabezado[0:20, 70:250]
    nombres.append(nombre)
    id = encabezado[0:20, 300:400]
    code = encabezado[0:20, 460:535]
    fecha = encabezado[0:20, 619:725]
    # Obtengo los datos de cada campo
    d_nombre = analizar_imagen_rasgos_letras(nombre)
    d_id = analizar_imagen_rasgos_letras(id)
    d_code = analizar_imagen_rasgos_letras(code)
    d_fecha = analizar_imagen_rasgos_letras(fecha)


# Leer la primera imagen para obtener sus dimensiones

    # Dimensiones de la imagen
import numpy as np
import matplotlib.pyplot as plt

# Crear la imagen con fondo blanco
image = np.ones((500, 400), dtype=np.uint8) * 255

y = 40
x = 40

for nombre in nombres:
    # nombre = nombres[0]
    nombre.shape
    image[x:x+20, y:y+180] = nombre

    # Definir la fuente, tamaño de fuente y color
    font = cv2.FONT_HERSHEY_SIMPLEX
    tamanio_fuente = 0.5
    color = (0, 0)  # Color en formato BGR (azul, verde, rojo)

    # Escribir el texto en la imagen compuesta
    imagen_con_texto = cv2.putText(image, "Aprobo", (y+250, x+20), font, tamanio_fuente, color, thickness=2)

    x += 100
    # Mostrar la imagen
plt.imshow(image, cmap='gray')
plt.show()






    # print("EXAMEN:", examen)
    # # Nombre
    # if d_nombre["Caracteres"] > 25 or d_nombre["Caracteres"] == 0 or  d_nombre["Palabras"] < 2:
    #     print("Nombre: Mal")
    # else:
    #     print("Nombre: Ok")
    
    # # Id
    # if d_id["Caracteres"] > 8 or d_id["Caracteres"] == 0 or d_id["Palabras"] > 1:
    #     print("Id: Mal")
    # else:
    #     print("Id: Ok")
    
    # # Code
    # if d_code["Caracteres"] == 1:
    #     print("Code: Ok")
    # else:
    #     print("Code: Mal")
    
    # # Fecha
    # if d_fecha["Caracteres"] > 8 or d_fecha["Caracteres"] == 0 or d_fecha["Palabras"] > 1:
    #     print("fecha: Mal")
    # else:
    #     print("fecha: Ok")



    


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



