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
    componentes_filtrados = []
    for i, stat in enumerate(stats):
        area = stat[4]
        height = stat[3]
        width = stat[2]

        # Filtrar por área mínima y relación de aspecto
        if area >= 300 and height < width:
            componentes_filtrados.append(i)
            
    componente_patente = stats[componentes_filtrados[-1]]
    coordenada_x = componente_patente[0] 
    coordenada_y = componente_patente[1]
    ancho  = componente_patente[2]
    largo  = componente_patente[3]
    patente = img[coordenada_y:coordenada_y + largo, coordenada_x: coordenada_x + ancho]
    # plt.imshow(patente, cmap='gray'), plt.show()
    return patente


# Imprimir 1 patente
img = cv2.imread('TP2/Patentes/img01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img), plt.show(block=False)
patente = recortar_patente(img)
plt.imshow(patente, cmap='gray'), plt.show()

# Imprimir todas las patentes
ruta_carpeta = "TP2/Patentes"
for nombre_archivo in os.listdir(ruta_carpeta):
    if os.path.isfile(os.path.join(ruta_carpeta, nombre_archivo)):
        print(os.path.join(ruta_carpeta, nombre_archivo))
        img_path = os.path.join(ruta_carpeta, nombre_archivo)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patente = recortar_patente(img)
        plt.imshow(patente, cmap='gray'), plt.show()

## SEGUNDA PARTE --> DETECTAR CARACTERES EN LA IMG RECORTADA

img = cv2.imread('TP2/Patentes/img01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img), plt.show(block=False)

patente1 = recortar_patente(img)
# plt.imshow(patente1, cmap='gray'), plt.show()

# Convierto la imagen a escala de grises
img_gris1 = cv2.cvtColor(patente1, cv2.COLOR_BGR2GRAY)
# plt.imshow(img_gris1, cmap='gray'), plt.show(block=False)

# La binarizo
_, img_binarizada = cv2.threshold(img_gris1, 149, 255, cv2.THRESH_BINARY)
# plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)

# Hago una clausura para terminar de cerrar los espacios
se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
fop = cv2.morphologyEx(img_binarizada, cv2.MORPH_CLOSE, se)
# plt.imshow(fop, cmap='gray'), plt.show()

img_copia = patente1.copy()
plt.imshow(img_copia), plt.show()

# Encontrar los componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fop, connectivity=8)

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)

for st in stats:
    if stats[4] < 500:
        cv2.rectangle(img_copia, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=1)

# Mostrar la imagen resultante
plt.imshow(img_copia)
plt.show()

# Dibujar los bounding boxes en los componentes detectados que cumplan con el criterio de área
for i in range(1, num_labels):  # Empezamos desde 1 para evitar el fondo
    # stats[i, 4] contiene el área del componente i-ésimo
    if  30 < stats[i, 4] < 500:
        cv2.rectangle(img_copia, (stats[i, 0], stats[i, 1]), 
                      (stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]), color=(0, 255, 0), thickness=1)
        
# Mostrar la imagen resultante
plt.imshow(cv2.cvtColor(img_copia, cv2.COLOR_BGR2RGB)), plt.show()

for i in range(1, num_labels):  # Ignoramos el primer componente (fondo)
    x, y, w, h, area = stats[i]
    if 100 < area < 500:  # Filtrar componentes por tamaño
        cv2.rectangle(img_copia, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen resultante
plt.imshow(img_copia)
plt.show()
