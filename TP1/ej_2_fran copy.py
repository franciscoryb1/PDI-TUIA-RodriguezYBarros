import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    
    salida = {
        "Caracteres": len(stats_filtrado),
        "Espacios": espacios_entre_letras,
        "Palabras": espacios_entre_letras + 1
    }

    return salida