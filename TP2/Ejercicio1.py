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

# Probamos aplicando apertura y luego un filtro gaussiano 
se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
fop = cv2.morphologyEx(img_gris, cv2.MORPH_OPEN, se)

# Aplico un filtro Gaussiano de suavizado. fui probando distintos tamaños de kernels y sigmaX 
blurred_img = cv2.GaussianBlur(fop, ksize=(3, 3), sigmaX=1.5)


plt.figure(figsize=(10, 5))

ax1 = plt.subplot(1, 2, 1)
plt.title('apertura')
plt.imshow(fop, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
plt.title('gaussiano')
plt.imshow(blurred_img, cmap='gray')
plt.axis('off')

plt.show()


# Aplico el algoritmo Canny para detectar bordes
edges1 = cv2.Canny(blurred_img, 0.04*255, 0.10*255)
edges2 = cv2.Canny(blurred_img, 0.35*255, 0.4*255)
edges3 = cv2.Canny(blurred_img, 0.20*255, 0.80*255)

# Muestro los distintos umbrales de canny
plt.figure(figsize=(10, 5))

ax2 = plt.subplot(1, 3, 1)
plt.title('Canny - U1:0.04% | U2:0.10%')
plt.imshow(edges1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.35% | U2:0.40%')
plt.imshow(edges2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3,  sharex=ax2, sharey=ax2)
plt.title('Canny - U1:0.20% | U2:0.75%')
plt.imshow(edges3, cmap='gray')
plt.axis('off')

plt.show()

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,7))
f_mg = cv2.morphologyEx(edges3, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg)

#Clausura
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
f_mg = cv2.morphologyEx(edges3, cv2.MORPH_CLOSE, kernel)
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
# --- Otra opcion ----------------
# output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
# # Resultados
# num_labels = output[0]  # Cantidad de elementos
# labels = output[1]      # Matriz con etiquetas
# stats = output[2]       # Matriz de stats
# centroids = output[3]   # Centroides de elementos
# --------------------------------

imshow(img=labels)

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
imshow(img=im_color, color_img=True)

#ESTO NO LO APLIQUE TODAVÍA PORQUE NO LOGRO QUE ME TOME EL CHIP ENTERO

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
# imshow(img=labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    if (300 <= st[2] <= 350) and (50<= st[3] <= 90):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
    if (80 <= st[2] <= 100) and (320<= st[3] <= 350):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=4)
    if (310 <= st[2] <= 330) and (580<= st[3] <= 600):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,0,255), thickness=4)
    if (170 <= st[2] <= 190) and (190<= st[3] <= 210):
        cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(255,0,0), thickness=4)  
imshow(img=im_color , color_img=True)             