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
img = cv2.imread('placa.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(); plt.imshow(img), plt.show(block=False)

# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplico un filtro Gaussiano de suavizado. fui probando distintos tamaños de kernels y sigmaX 
blurred_img = cv2.GaussianBlur(img_gris, ksize=(5, 5), sigmaX=1)
#plt.figure(), plt.imshow(blurred_img, cmap="gray"), plt.show(block=False)

# Aplico el algoritmo Canny para detectar bordes
edges = cv2.Canny(blurred_img, 100, 200)

#Gradiente morfológico
se = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
f_mg = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, se)
imshow(f_mg)

# Muestro la imagen con suavizado, la imagen con los bordes detectados y gradiente morfológico
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Imagen con suavizado')
plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Bordes Detectados')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Gradiente morfológico')
plt.imshow(f_mg, cmap='gray')
plt.axis('off')

plt.show()

#HASTA ACÁ VA BIEN, DESPUES EMPIEZO A HACER CUALQUIERA

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
for centroid in centroids:
cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
imshow(img=im_color, color_img=True)

# Definir las coordenadas del bounding box para el chip
x, y, width, height = 50, 50, 200, 150

# Bounding Box
cnt = contours[12]
x,y,w,h = cv2.boundingRect(cnt)
f = cv2.imread('contornos.png')
cv2.drawContours(f, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.rectangle(f, (x,y), (x+w,y+h), color=(255, 0, 255), thickness=2)
cv2.imshow('boundingRect', f)

# Mostrar la imagen con el bounding box
cv2.imshow('Imagen con Bounding Box', img_chip)
cv2.waitKey(0)
cv2.destroyAllWindows()
