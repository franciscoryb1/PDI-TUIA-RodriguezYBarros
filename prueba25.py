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

cap = cv2.VideoCapture('TP3/ruta_1.mp4')  # Abro el video
#cap = cv2.VideoCapture('TP3/ruta_2.mp4')  # Abro el video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

ret, frame = cap.read()
imshow(frame)

# Crear una máscara en blanco
mask = np.zeros((height, width), dtype=np.uint8)

# Definir los puntos del triángulo (en este caso, una diagonal)
height, width = frame.shape[:2]
points = np.array([[115, height], [915, height], [560, 330], [400, 330] ], dtype=np.int32)

# Rellenar el triángulo en la máscara
cv2.fillPoly(mask, [points], 255)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(frame, frame, mask=mask)

# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# La binarizo
_, img_binarizada = cv2.threshold(img_gris, 200, 255, cv2.THRESH_BINARY)

# Canny
edges1 = cv2.Canny(img_binarizada, 0.2*255, 0.60*255)

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
f_mg = cv2.morphologyEx(edges1, cv2.MORPH_GRADIENT, kernel)

#clausura
clau = cv2.morphologyEx(f_mg, cv2.MORPH_CLOSE, kernel)

imshow(clau)

Rho = 1                   # rho: resolución de la distancia en píxeles
Theta = np.pi/180       # theta: resolución del ángulo en radianes
Threshold = 100             # threshold: número mínimo de intersecciones para detectar una línea
minLineLength = 50          # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
maxLineGap = 10             # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea
# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rho ,Theta,Threshold,minLineLength,maxLineGap)

# Convertir las líneas en una lista de listas de tuplas
lines = [line[0] for line in lines]

# Ordenar por coordenada x inicial
lines = sorted(lines, key=lambda line: line[0])

lines[1]

merged_lines = []
current_line = lines[0]

for next_line in lines[1:]:
    # Verificar si el siguiente segmento está lo suficientemente cerca para ser combinado
    if (abs(next_line[0] - current_line[2]) <= umbral_x) and (abs(next_line[1] - current_line[3]) <=umbral_y):
        # Combinar los segmentos actualizando la coordenada final del segmento actual
        current_line = [current_line[0], current_line[1], next_line[2], next_line[3]]
    else:
        # Agregar la línea actual a las líneas combinadas y actualizar la línea actual
        merged_lines.append(current_line)
        current_line = next_line

# Agregar la última línea
merged_lines.append(current_line)

return merged_lines


lines[25]

final = frame.copy()

# Suponiendo que lines es una lista de segmentos de líneas detectadas
# Cada línea tiene la forma [x1, y1, x2, y2]

# lines es la lista de líneas detectadas
merged_lines = unir_segmentos(lines, umbral_x=70, umbral_y= 20)

# Dibujar las líneas detectadas
for linea in merged_lines:
    x1, y1, x2, y2 = linea
    cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 7)
return final