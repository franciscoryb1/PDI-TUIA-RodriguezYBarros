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
img = cv2.imread('TP2/placa.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(); plt.imshow(img), plt.show(block=False)

# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Aplico un filtro
blurred_img = cv2.medianBlur(img_gris, ksize=5)

# Aplico el algoritmo Canny para detectar bordes
edges1 = cv2.Canny(blurred_img, 0.04*255, 0.10*255)
edges2 = cv2.Canny(blurred_img, 0.35*255, 0.4*255)
edges3 = cv2.Canny(blurred_img, 0.20*255, 0.80*255) # ESTE SE USA PARA CAPACITORES Y CHIP
edges4 = cv2.Canny(blurred_img, 0.20*255, 0.60*255)
edges5 = cv2.Canny(blurred_img, 0.20*255, 0.40*255)

# Muestro los distintos umbrales de canny
plt.figure(figsize=(10, 5))

ax2 = plt.subplot(1, 5, 1)
plt.title('Canny - U1:0.04% | U2:0.10%')
plt.imshow(edges1, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 2,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.35% | U2:0.40%')
plt.imshow(edges2, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.80%')
plt.imshow(edges3, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 4,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.60%')
plt.imshow(edges4, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 5,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.40%')
plt.imshow(edges5, cmap='gray')
plt.axis('off')
plt.show()

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
f_mg = cv2.morphologyEx(edges3, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg)

# Muestro la imagen con suavizado, la imagen con los bordes detectados y gradiente morfológico
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Imagen con suavizado')
plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Bordes Detectados')
plt.imshow(edges3, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Gradiente morfológico')
plt.imshow(f_mg, cmap='gray')
plt.axis('off')

plt.show()

#
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_mg, connectivity, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
imshow(img=im_color, color_img=True)

## Para el chip y los capacitores vamos a utilizar los mismos umbrales de la img, 
# pero para las resistencias vamos a utilizar otros, por lo que primero vamos a detectar el chip 
# y los capacitores, rellenamos con blanco en esa parte de la img y luego cambiamos los umbrales 
# para poder detectar las resistencias

# Detectamos el chip

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    if (285 <= st[2] <= 315) and (590<= st[3] <= 610):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
imshow(img=im_color , color_img=True) 


# Detectamos los capacitores 

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    if (490 <= st[2] <= 520) and (600<= st[3] <= 650):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
        #cv2.putText(image, "Grande", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if (330 <= st[2] <= 360) and (300<= st[3] <= 340):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,0,255), thickness=4)
    if (300 <= st[2] <= 330) and (360<= st[3] <= 400):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,0,255), thickness=4)
    if (150 <= st[2] <= 170) and (150<= st[3] <= 210):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(255,0,0), thickness=4)
    if (250 <= st[2] <= 266) and (150<= st[3] <= 210):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(255,0,0), thickness=4)  
    if (180 <= st[2] <= 210) and (230<= st[3] <= 250):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,255), thickness=4) 
    if (260 <= st[2] <= 270) and (235<= st[3] <= 245):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,255), thickness=4) 
imshow(img=im_color , color_img=True) 

# ESTO ES PARA EL EJERCICIO B
print(f"Cantidad de resistencias muy pequeñas: {small_count}")
print(f"Cantidad de resistencias pequeñas: {small_count}")
print(f"Cantidad de resistencias medianas: {medium_count}")
print(f"Cantidad de resistencias grandes: {large_count}")

#ESTO LO HACEMOS PARA DESPUES PODER DETECTAR MEJOR LAS RESISTENCIAS
# Crear una copia de la imagen original para pintar los capacitores y el chip
original_img = cv2.imread('TP2/placa.png')  # Cambia esta ruta a la ruta de tu imagen original

# Listas para las regiones a pintar de negro
regiones_a_pintar = []

for st in stats:
    # Rellena los sectores de los capacitores con negro 
    if (490 <= st[2] <= 520) and (600 <= st[3] <= 650):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (330 <= st[2] <= 360) and (300 <= st[3] <= 340):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 0, 255), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (300 <= st[2] <= 330) and (360 <= st[3] <= 400):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 0, 255), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (150 <= st[2] <= 170) and (150 <= st[3] <= 210):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(255, 0, 0), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (250 <= st[2] <= 266) and (150 <= st[3] <= 210):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(255, 0, 0), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (180 <= st[2] <= 210) and (230 <= st[3] <= 250):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 255), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    if (260 <= st[2] <= 270) and (235 <= st[3] <= 245):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 255), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))
    # Rellena el sector del chip con negro
    if (285 <= st[2] <= 315) and (590 <= st[3] <= 610):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=4)
        regiones_a_pintar.append((st[0], st[1], st[2], st[3]))

# Pintar las ROI de negro en la imagen original
for x, y, w, h in regiones_a_pintar:
    original_img[y:y+h, x:x+w] = 0

# Guardar la imagen modificada
cv2.imwrite('TP2/imagen_con_capacitores_y_chip_negros.png', original_img)

# Mostrar la imagen con los capacitores pintados de negro usando la función imshow
imshow(img=original_img, color_img=True, title="Imagen con capacitores y chip pintados de negro")

#VAMOS A CREAR UNA IMAGEN CON EL CHIP, OTRA CON LOS CAPACITORES Y OTRA CON AMBAS COSAS
# Crear una copia de la imagen original para pintar las ROI
original_img = cv2.imread('TP2/placa.png')  

# Crear una imagen negra de las mismas dimensiones que la imagen original
img_negra = np.zeros_like(original_img)
img_negra1 = np.zeros_like(original_img)

# Contador para los nombres de archivo
contador = 0
regiones_a_pintar = []

for st in stats:
    x, y, w, h = st[0], st[1], st[2], st[3]
    # Filtrar según las condiciones dadas
    if ((490 <= w <= 520) and (600 <= h <= 650)) or \
       ((330 <= w <= 360) and (300 <= h <= 340)) or \
       ((300 <= w <= 330) and (360 <= h <= 400)) or \
       ((150 <= w <= 170) and (150 <= h <= 210)) or \
       ((250 <= w <= 266) and (150 <= h <= 210)) or \
       ((180 <= w <= 210) and (230 <= h <= 250)) or \
       ((260 <= w <= 270) and (235 <= h <= 245)):# or \
       #((285 <= w <= 315) and (590 <= h <= 610)):
        regiones_a_pintar.append((x, y, w, h))

# Recortar y guardar cada elemento detectado y colocarlos en la imagen negra
for (x, y, w, h) in regiones_a_pintar:
    # Recortar el elemento de la imagen original
    elemento = original_img[y:y+h, x:x+w]

    # # Guardar la imagen del elemento
    # cv2.imwrite(f'elementos/elemento_{contador}.png', elemento)
    # imshow(img=elemento, color_img=True, title="Elemento{contador}")

    # Colocar el elemento en la imagen negra en su posición correspondiente
    img_negra[y:y+h, x:x+w] = elemento
    img_negra1[y:y+h, x:x+w] = elemento

    # Incrementar el contador
   # contador += 1

# Guardar la imagen negra con los elementos en sus posiciones
cv2.imwrite('TP2/imagen_con_componentes.png', img_negra)
cv2.imwrite('TP2/capacitores.png', img_negra)

imshow(img=img_negra, color_img=True, title="Imagen negra con elementos en sus posiciones")

# Crear una copia de la imagen original para pintar las ROI
original_img = cv2.imread('TP2/placa.png') 

# Crear una imagen negra de las mismas dimensiones que la imagen original
img_negra2 = np.zeros_like(original_img)

#vamos a crear una imagen solo con el chip
regiones_a_pintar = []
for st in stats:
    x, y, w, h = st[0], st[1], st[2], st[3]
    # Filtrar según las condiciones dadas
    if ((285 <= w <= 315) and (590 <= h <= 610)):
        regiones_a_pintar.append((x, y, w, h))

# Recortar y guardar cada elemento detectado y colocarlos en la imagen negra
for (x, y, w, h) in regiones_a_pintar:
    # Recortar el elemento de la imagen original
    elemento = original_img[y:y+h, x:x+w]

    # Colocar el elemento en la imagen negra en su posición correspondiente
    img_negra[y:y+h, x:x+w] = elemento
    img_negra2[y:y+h, x:x+w] = elemento

# Guardar la imagen negra con los elementos en sus posiciones
cv2.imwrite('TP/chip.png', img_negra2)
cv2.imwrite('TP2/imagen_con_componentes.png', img_negra)

imshow(img=img_negra, color_img=True, title="chip")



# REPETIMOS LO QUE HICIMOS ANTES PERO CON LA IMG NUEVA QUE TIENE LOS CAPACITORES Y EL CHIP PINTADOS DE NEGRO 

# Cargo Imagen
img2 = cv2.imread('TP2/imagen_con_capacitores_y_chip_negros.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.figure(); plt.imshow(img2), plt.show(block=False)

# Convierto la imagen a escala de grises
img_gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#binarizamos la imagen para resaltar las resistencias
_, img_binarizada = cv2.threshold(img_gris2, 130, 255, cv2.THRESH_BINARY)
plt.figure(); plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)


# Aplico un filtro Gaussiano de suavizado. fui probando distintos tamaños de kernels y sigmaX 
blurred_img2 = cv2.medianBlur(img_binarizada, ksize=5)
plt.figure(); plt.imshow(blurred_img2, cmap='gray'), plt.show(block=False)

# Aplico el algoritmo Canny para detectar bordes
edges1_2 = cv2.Canny(blurred_img2, 0.50*255, 0.18*255)
edges2_2 = cv2.Canny(blurred_img2, 0.35*255, 0.4*255)
edges3_2 = cv2.Canny(blurred_img2, 0.20*255, 0.80*255) # ESTE SE USA PARA CAPACITORES Y CHIP
edges4_2 = cv2.Canny(blurred_img2, 0.20*255, 0.60*255)
edges5_2 = cv2.Canny(blurred_img2, 0.20*255, 0.40*255)

# Muestro los distintos umbrales de canny
plt.figure(figsize=(10, 5))

ax2 = plt.subplot(2, 3, 1)
plt.title('Canny - U1:0.04% | U2:0.10%')
plt.imshow(edges1_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.35% | U2:0.40%')
plt.imshow(edges2_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.80%')
plt.imshow(edges3_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.60%')
plt.imshow(edges4_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.40%')
plt.imshow(edges5_2, cmap='gray')
plt.axis('off')
plt.show()

#Clausura
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
c_img2 = cv2.morphologyEx(edges4_2, cv2.MORPH_CLOSE, kernel)
imshow(c_img2)

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
f_mg2 = cv2.morphologyEx(c_img2, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg2)

connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_mg2, connectivity, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color2 = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color2, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color2, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
imshow(img=im_color2, color_img=True)

# Detectamos las resistencias 

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    if (60 <= st[2] <= 80) and (260<= st[3] <= 290):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
    if (240 <= st[2] <= 300) and (60<= st[3] <= 80):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4) 
    if (240 <= st[2] <= 300) and (100<= st[3] <= 120):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
imshow(img=im_color , color_img=True) 


#agregamos las resistencias a la imagen completa que tiene fondo negro y el chip y los capacitores 

# Crear una imagen negra de las mismas dimensiones que la imagen original
img_negra3 = np.zeros_like(original_img)

# Contador para la cantidad de resistencias
contador = 0
regiones_a_pintar = []

for st in stats:
    x, y, w, h = st[0], st[1], st[2], st[3]
    # Filtrar según las condiciones dadas
    if ((60 <= w <= 80) and (260<= h <= 290)) or \
       ((240 <= w <= 300) and (60<= h <= 80)) or \
       ((240 <= w <= 300) and (100<= h <= 120)):
        regiones_a_pintar.append((x, y, w, h))

# Recortar y guardar cada elemento detectado y colocarlos en la imagen negra
for (x, y, w, h) in regiones_a_pintar:
    # Recortar el elemento de la imagen original
    elemento = original_img[y:y+h, x:x+w]

    # Colocar el elemento en la imagen negra en su posición correspondiente
    img_negra[y:y+h, x:x+w] = elemento
    img_negra3[y:y+h, x:x+w] = elemento

    # Incrementar el contador
    contador += 1
print(f"Cantidad de resistencias eléctricas: {contador}")

# Guardar la imagen negra con los elementos en sus posiciones
cv2.imwrite('TP2/imagen_con_componentes.png', img_negra)
cv2.imwrite('TP2/resistencias.png', img_negra3)

#imagen final con chip, capacitores y resistencias
imshow(img=img_negra, color_img=True, title="Imagen negra con elementos en sus posiciones")



# Mostrar los resultados por consola
print(f"Cantidad de resistencias muy pequeñas: {small_count}")
print(f"Cantidad de resistencias pequeñas: {small_count}")
print(f"Cantidad de resistencias medianas: {medium_count}")
print(f"Cantidad de resistencias grandes: {large_count}")

