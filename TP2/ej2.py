import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def recortar_patente(img):
# Convierto la imagen a escala de grises
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure(); plt.imshow(img_gris, cmap='gray'), plt.show(block=False)

    # Aplicar un filtro Gaussiano para reducir el ruido
    # blurred = cv2.GaussianBlur(img_gris, (3, 3), 0)
    # plt.figure(); plt.imshow(blurred, cmap='gray'), plt.show(block=False)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(img_gris, (1, 21), 0)
    # plt.figure(); plt.imshow(blurred, cmap='gray'), plt.show(block=False)
    # canny
    img_canny = cv2.Canny(blurred, 250, 300)
    # plt.imshow(img_canny, cmap='gray'), plt.show()
    # cierre
    elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)
    # plt.imshow(img_cierre, cmap='gray'), plt.show()
    # componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cierre)
    area_minima_componente = 300
    componentes_filtrados = []
    for i, stat in enumerate(stats):
        area = stat[4]
        height = stat[3]
        width = stat[2]
        # Filtrar por área mínima y relación de aspecto
        if area >= area_minima_componente and height < width:
            componentes_filtrados.append(i)
    componente_patente = stats[componentes_filtrados[-1]]
    coordenada_x = componente_patente[0] 
    coordenada_y = componente_patente[1]
    ancho  = componente_patente[2]
    largo  = componente_patente[3]
    patente = img[coordenada_y:coordenada_y + largo, coordenada_x: coordenada_x + ancho]
    # plt.imshow(patente, cmap='gray'), plt.show()
    return patente

# Cargo Imagen
img = cv2.imread('TP2/Patentes/img01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(); plt.imshow(img), plt.show(block=False)
patente = recortar_patente(img)
plt.imshow(patente, cmap='gray'), plt.show()





# # Ruta de la carpeta que contiene los archivos de patentes
# ruta_carpeta = "TP2/Patentes"
# # Recorrer todos los archivos en la carpeta
# for nombre_archivo in os.listdir(ruta_carpeta):
#     # Verificar si es un archivo
#     if os.path.isfile(os.path.join(ruta_carpeta, nombre_archivo)):
#         # Hacer algo con el nombre del archivo o su ruta completa
#         print(os.path.join(ruta_carpeta, nombre_archivo))
#         img_path = os.path.join(ruta_carpeta, nombre_archivo)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         patente = recortar_patente(img)
#         plt.imshow(patente, cmap='gray'), plt.show()