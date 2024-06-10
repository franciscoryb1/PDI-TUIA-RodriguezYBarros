import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Cargo Imagen
img = cv2.imread('TP2/Patentes/img11.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(); plt.imshow(img), plt.show(block=False)

# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(); plt.imshow(img_gris, cmap='gray'), plt.show(block=False)

# Aplicar un filtro Gaussiano para reducir el ruido
# blurred = cv2.GaussianBlur(img_gris, (3, 3), 0)
# plt.figure(); plt.imshow(blurred, cmap='gray'), plt.show(block=False)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(img_gris, (5, 5), 0)
plt.figure(); plt.imshow(blurred, cmap='gray'), plt.show(block=False)

# img_binarizada = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# plt.figure(); plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)

_, img_binarizada = cv2.threshold(img_gris, 128, 255, cv2.THRESH_BINARY)
plt.figure(); plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)
np.unique(img_binarizada)


# Encontrar contornos en la imagen umbralizada
contours, hierarchy = cv2.findContours(img_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en una imagen en blanco
contour_img = np.zeros_like(img_binarizada)
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
plt.figure(); plt.imshow(contour_img, cmap='gray'), plt.show(block=False)

# Detecta los componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contour_img, 8, cv2.CV_32S)
stats = sorted(stats, key=lambda x: x[0])

# x: Coordenada x del borde izquierdo del rectángulo delimitador del componente. (stats[:, 0])
# y: Coordenada y del borde superior del rectángulo delimitador del componente. (stats[:, 1])
# width: Ancho del rectángulo delimitador del componente. (stats[:, 2])
# height: Altura del rectángulo delimitador del componente. (stats[:, 3])
# area: Área del componente en píxeles. (stats[:, 4])

# Crear una nueva imagen en negro para los componentes filtrados
filtered_img = np.zeros(contour_img.shape, dtype=np.uint8)

# Crear una lista para almacenar los puntos iniciales y los centroides de los componentes filtrados
filtered_components = []

# Recorrer cada contorno y filtrar por área 
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    # x, y, width, height, area = stats[contour]
    area = cv2.contourArea(contour)
    
    # Ajustar estos parámetros según sea necesario
    if 2000 < area < 2300 :
        cv2.drawContours(filtered_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Mostrar la imagen con los componentes filtrados
plt.figure(); plt.imshow(filtered_img, cmap='gray'), plt.show(block=False)




# Crear una nueva imagen en negro para los componentes filtrados
filtered_img = np.zeros(contour_img.shape, dtype=np.uint8)

# Crear una lista para almacenar los puntos iniciales y los centroides de los componentes filtrados
filtered_components = []

# Recorrer cada componente conectado
for i in range(1, num_labels):  # Saltamos la etiqueta 0, que es el fondo
    # Obtener las estadísticas del componente
    x, y, width, height, area = stats[i]

    # Si el área es igual a 2100, añadir el componente a la imagen filtrada
    if 2000 < area < 2300:
        filtered_img[labels == i] = 255
        # Punto inicial (superior izquierdo)
        top_left_x = x
        top_left_y = y
        # Centroide
        centroid_x, centroid_y = centroids[i]
        # filtered_img[patente == 255] = 255

        # Añadir a la lista de componentes filtrados
        filtered_components.append({
            'label': i,
            'top_left': (top_left_x, top_left_y),
            'centroid': (centroid_x, centroid_y)
        })

plt.figure(); plt.imshow(filtered_img, cmap='gray'), plt.show(block=False)

semilla = filtered_components[0]['top_left']

componente_interes = np.uint8(labels == 194) * 255

# Encuentra todos los puntos blancos en la imagen del componente de interés
puntos_blancos = cv2.findNonZero(componente_interes)

# Itera sobre cada punto blanco y obtén sus coordenadas
coordenadas = []
for punto in puntos_blancos:
    x, y = punto[0]  # El punto es una matriz de una fila con dos elementos (x, y)
    coordenadas.append((x, y))

# Ahora puedes pintar estos puntos de 255 en otra imagen si lo deseas
imagen_pintada = np.zeros_like(componente_interes)  # Crea una imagen en blanco del mismo tamaño que tu componente de interés
# ---------------------------------------------------------------------------------------
# --- Reconstrucción Morgológica --------------------------------------------------------
# ---------------------------------------------------------------------------------------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

img_r = imreconstruct(marker=semilla, mask=filtered_img)  

## No conviene aplicar apertura, ni filtro gaussiano, ni erosion 

edges3 = cv2.Canny(img_binarizada, 0.60*255, 0.80*255)
plt.figure(); plt.imshow(edges3, cmap='gray'), plt.show(block=False)


####################################################################################################################################################
####################################################################################################################################################
# ---------------------------------------------------------------------------------------
# --- Reconstrucción Morgológica --------------------------------------------------------
# ---------------------------------------------------------------------------------------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

# --- Version 1 ------------------------------------------------
# Utilizando reconstrucción morfológica
# NO rellena los huecos que tocan los bordes
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

# --- Mejoras sobre la imagen original ----------
# img = cv2.dilate(img, np.ones((3,3),np.uint8))  # Con esto logro rellenar casi todos, algunas que quedan son: 1 "b", 1 "o" y 1 "e" abajo)
# img = cv2.dilate(img, np.ones((5,5),np.uint8))  # Soluciono solo la "e" 
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) # Notar la mejora en el grosor...
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8)) # Acá ya hay problemas: se unen caracteres.
# -----------------------------------------------
img_fh = imfillhole(img)
plt.figure()
ax1 = plt.subplot(121); imshow(img, new_fig=False, title="Original", ticks=True)
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(img_fh, new_fig=False, title="Rellenado de Huecos")
plt.show(block=False)

# --- Analisis de cada etapa ------------------------------------
img = cv2.imread('book_text_bw.tif', cv2.IMREAD_GRAYSCALE)
mask = np.zeros_like(img)                                                   # Genero mascara para...
mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.

plt.figure()
ax1 = plt.subplot(221); imshow(marker, new_fig=False, title="Marker", ticks=True)
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(img_c, new_fig=False, title="Mascara")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(img_r, new_fig=False, title="Reconstruccion")
plt.subplot(224, sharex=ax1, sharey=ax1); imshow(img_fh, new_fig=False, title="Reconstruccion + Complemento")
plt.show(block=False)
####################################################################################################################################################
####################################################################################################################################################

# Cargar la imagen
image_path = 'TP2/Patentes/img01.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plt.figure(); plt.imshow(blurred, cmap='gray'), plt.show(block=False)

# Aplicar la detección de bordes Canny
edges = cv2.Canny(blurred, 50, 150)
plt.figure(); plt.imshow(edges, cmap='gray'), plt.show(block=False)

# Encontrar los contornos en la imagen de bordes
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar los contornos para encontrar la patente
for contour in contours:
    # Aproximar el contorno
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Los contornos de la patente suelen ser rectángulos
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6:  # La relación de aspecto típica de una patente
            if 1000 < cv2.contourArea(contour) < 5000:  # Filtrar por área
                # Dibujar el contorno encontrado
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                # Guardar la imagen de la patente
                plate = gray[y:y+h, x:x+w]
                cv2.imwrite('/mnt/data/patente_detectada.png', plate)
                break

# Mostrar la imagen original con el contorno de la patente
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Imagen con patente detectada')
plt.axis('off')
plt.show()