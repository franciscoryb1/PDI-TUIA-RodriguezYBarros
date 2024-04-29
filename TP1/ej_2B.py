import cv2
import numpy as np
import matplotlib.pyplot as plt


def analizar_imagen_rasgos_letras(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Encontrar contornos en la imagen umbralizada
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos en una imagen en blanco
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Detecta los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    stats = sorted(stats, key=lambda x: x[0])

    # Filtra los componentes cuya área es menor a 50 pixeles
    stats_filtrado = [s for s in stats if s[-1] < 50]

    umbral_max = 9
    espacios_entre_letras = 0

    repetidos = []

    for i in range(len(stats_filtrado)-1):
        # Obtiene las coordenadas x del componente actual y del siguiente
        x_actual = stats_filtrado[i][0]
        x_siguiente = stats_filtrado[i + 1][0]
        # Calcula la distancia horizontal entre los componentes
        distancia_horizontal = x_siguiente - x_actual

        # Si la distancia horizontal es mayor que cierto umbral intuimos que hay un espacio entre palabras.
        if distancia_horizontal >= umbral_max:
            # Incrementa el contador de espacios entre letras
            espacios_entre_letras += 1

    # Elimina los elementos de "stats_filtrados" que están en "repetidos"
    stats_filtrado = [arr for arr in stats_filtrado if not any((arr == elem).all() for elem in repetidos)]
    # if len(stats_filtrado) < 1:
    #     espacios_entre_letras = -1
    salida = {
        "Caracteres": len(stats_filtrado),
        "Espacios": espacios_entre_letras,
        "Palabras": espacios_entre_letras + 1
    }

    return salida

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
    imagen_con_texto = cv2.putText(image, 'Aprobado', (y+250, x+20), font, tamanio_fuente, color, thickness=2)

    x += 100
    # Mostrar la imagen
plt.imshow(image, cmap='gray')
plt.show()

# 1- APROBADO
# 2- APROBADO
# 3- NO APROBADO
# 4- NO APROBADO
# 5- NO APROBADO







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



