import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

# -------------------------------------------------------------------
# --- Análisis Morfológico en Escala de Grises ----------------------
# -------------------------------------------------------------------

# --- Cargo Imagen --------------------------------------------------
f = cv2.imread('city.tif', cv2.IMREAD_GRAYSCALE)
f.shape
np.unique(f)
imshow(f)

# --- Dilatacion, Erosion y Gradiente Morfologico -------------------
k = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
fd = cv2.dilate(f, kernel)
fe = cv2.erode(f, kernel)
fmg = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, kernel)
np.unique(fd)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fd, cmap='gray'), plt.title('Dilatacion'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fe, cmap='gray'), plt.title('Erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(fmg, cmap='gray'), plt.title('Gradiente Morfologico'), plt.xticks([]), plt.yticks([])
plt.show(block=False)

# --- Apertura y Clausura -------------------------------------------
f = cv2.imread('tacos_madera.tif', cv2.IMREAD_GRAYSCALE)
f.shape
np.unique(f)
imshow(f)

k = 9
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
fo = cv2.morphologyEx(f, cv2.MORPH_OPEN, kernel)
fc = cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fo, cmap='gray'), plt.title('Apertura'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fc, cmap='gray'), plt.title('Clausura'), plt.xticks([]), plt.yticks([])
plt.show(block=False)

# --- Operaciones alternantes ---------------------------------------
fasf = f.copy()
for k in np.arange(2,5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    fasf = cv2.morphologyEx( cv2.morphologyEx(fasf, cv2.MORPH_OPEN, kernel) , cv2.MORPH_CLOSE, kernel)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fo, cmap='gray'), plt.title('Apertura'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fc, cmap='gray'), plt.title('Clausura'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(fasf, cmap='gray'), plt.title('Alternante'), plt.xticks([]), plt.yticks([])
plt.show(block=False)

# --- Top-hat -------------------------------------------------------
f = cv2.imread('rice.tif', cv2.IMREAD_GRAYSCALE)
imshow(f, title="Imagen Original")

# Umbralado global
umbral, g1 = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)     # Similar a Matlab: g1=im2bw(f);
imshow(g1,title="Umbralado global")

# Top-Hat - Calculo Manual
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  
fo = cv2.morphologyEx(f, kernel=se, op=cv2.MORPH_OPEN)
imshow(fo, title="Modelo el fondo --> Apertura con kernel grande")
g2 = cv2.absdiff(f, fo)     
imshow(g2, title="Top-Hat - Calculo Manual")

# Top-Hat - OpenCV
g3 = cv2.morphologyEx(f, kernel=se, op=cv2.MORPH_TOPHAT)
imshow(g3, title="Top-Hat - OpenCV")

# Umbralado
umbral, g4 = cv2.threshold(g2, 50, 255, cv2.THRESH_BINARY)  # https://docs.opencv.org/3.4.15/db/d8e/tutorial_threshold.html   /   https://docs.opencv.org/3.4.15/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
imshow(g4, title="Top-Hat + Umbralado global")

# Visualizo
plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fo, cmap='gray'), plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(g2, cmap='gray'), plt.title('Imagen Original - Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(g3, cmap='gray'), plt.title('Top-hat'), plt.xticks([]), plt.yticks([])
plt.show(block=False)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(g1, cmap='gray'), plt.title('Umbralado Global'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(g3, cmap='gray'), plt.title('Top-hat'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(g4, cmap='gray'), plt.title('Top-hat + Umbralado Global'), plt.xticks([]), plt.yticks([])
plt.show(block=False)


