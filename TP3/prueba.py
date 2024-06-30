import cv2
import matplotlib.pyplot as plt
import numpy as np

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

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('TP3/ruta_1.mp4')                # Abro el video
# cap = cv2.VideoCapture('TP3/ruta_2.mp4')                # Abro el video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # No la usamos en este script,...
fps = int(cap.get(cv2.CAP_PROP_FPS))                # ... pero puede ser útil en otras ocasiones
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #
ret, frame = cap.read()
cap.release() 

# Crear una máscara en blanco
mask = np.zeros((height, width), dtype=np.uint8)

# Definir los puntos del triángulo (en este caso, una diagonal)
height, width = frame.shape[:2]
points = np.array([[115, height], [915, height], [560, 330], [400, 330] ], dtype=np.int32)

# Rellenar el triángulo en la máscara
cv2.fillPoly(mask, [points], 255)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(frame, frame, mask=mask)

# Mostrar la imagen original y la imagen resultante
cv2.imshow('Original', frame)
cv2.imshow('Cortada en diagonal', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
imshow(img_gris)

# La binarizo
_, img_binarizada = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)
plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)

# Canny
edges1 = cv2.Canny(img_binarizada, 0.2*255, 0.60*255)

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
f_mg = cv2.morphologyEx(edges1, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_mg, 8, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

stats_filtrado = stats[stats[:, 0] != 0]

valor = float('inf')
registro = None
for i, stat in enumerate(stats_filtrado):
    if stat[0] < valor:
        valor = stat[0]
        registro = i

x, y, ancho, alto, area = stats_filtrado[i] 

# Calcular la pendiente (m) y la intersección (b) de la línea
m = (alto) / (ancho)
b = (y+alto) - m * x

# Definir una función para la ecuación de la línea
def ecuacion_linea(x):
    return m * x + b

Rres = 1
Thetares = np.pi/180
Threshold = 1
minLineLength = 1
maxLineGap = 5
# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rres,Thetares,Threshold,minLineLength,maxLineGap)

# Dibujar las líneas detectadas
for linea in lines:
    x1, y1, x2, y2 = linea[0]
    cv2.line(f_mg, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Dibujar la línea calculada
x_vals = np.array(range(0, f_mg.shape[1]))
y_vals = ecuacion_linea(x_vals)
for i in range(len(x_vals)-1):
    cv2.line(f_mg, (x_vals[i], y_vals[i]), (x_vals[i+1], y_vals[i+1]), (255, 0, 0), 2)

# Mostrar la imagen resultante
plt.imshow(cv2.cvtColor(f_mg, cv2.COLOR_BGR2RGB))
plt.show()


Rres = 1
Thetares = np.pi/180
Threshold = 1
minLineLength = 1
maxLineGap = 5
# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rres,Thetares,Threshold,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(img_binarizada, )

final = frame.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]  # Obtener los puntos extremos de la línea
    cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar la línea sobre la imagen original
imshow(final)
# rho: resolución de la distancia en píxeles
# theta: resolución del ángulo en radianes
# threshold: número mínimo de intersecciones para detectar una línea
# minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
# maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea

## HAY QUE SACAR LA PENDIENTE Y QUE EN CADA FRAME VAYA CALCULANDO LA PENDIENTE Y DIBUJANDO LA LINEA


while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True: 
                                                              # ret indica si la lectura fue exitosa (True) o no (False)
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))  # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas
