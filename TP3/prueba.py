import cv2
import matplotlib.pyplot as plt
import numpy as np

# Defino función para mostrar imágenes
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


def drawlines(frame, width, height):
    # Creo una máscara en blanco
    mask = np.zeros((height, width), dtype=np.uint8)

    # Defino los puntos del trapecio (en este caso, una diagonal)
    height, width = frame.shape[:2]
    points = np.array([[115, height], [915, height], [560, 330], [400, 330] ], dtype=np.int32)

    # Relleno el triángulo en la máscara
    cv2.fillPoly(mask, [points], 255)

    # Aplico la máscara a la imagen original
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
    Rho = 1                     # rho: resolución de la distancia en píxeles
    Theta = np.pi/180           # theta: resolución del ángulo en radianes
    Threshold = 100             # threshold: número mínimo de intersecciones para detectar una línea
    minLineLength = 50          # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
    maxLineGap = 10             # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea
    # Aplico la transformada de Hough probabilística
    lines = cv2.HoughLinesP(f_mg, Rho ,Theta,Threshold,minLineLength,maxLineGap)

    # Convierto las líneas en una lista de listas de tuplas
    lines = [line[0] for line in lines]

    # Ordeno por coordenada x inicial
    lines = sorted(lines, key=lambda line: line[0])
    
    lineas_izq = []
    lineas_der = []
    for line in lines:
        if line[0] < 500:
            lineas_izq.append(line)
        else:
            lineas_der.append(line)

    x1, y1 = lineas_izq[0][0], lineas_izq[0][1]
    lineas_izq = sorted(lineas_izq, key=lambda lineas_izq: lineas_izq[3])
    x2, y2 = lineas_izq[0][2], lineas_izq[0][3]

    x1d, y1d = lineas_der[0][0], lineas_der[0][1]
    x2d, y2d = lineas_der[-1][2], lineas_der[-1][3]
    
    #creo una copia para no pisar el original
    final = frame.copy()

    # Dibujo las líneas detectadas
    cv2.line(final, (x1, y1), (x2, y2), (255, 0, 0), 6)
    cv2.line(final, (x1d, y1d), (x2d, y2d), (255, 0, 0), 6)
    return final


cap = cv2.VideoCapture('TP3/ruta_1.mp4')  # Abro el video
#cap = cv2.VideoCapture('TP3/ruta_2.mp4')  # Abro el video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #

while cap.isOpened():  # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()  # Obtengo el frame
    if ret: 
        # Asegurarse de que el frame está en el formato correcto
        if frame is None:
            break
        frame_l = drawlines(frame, frame.shape[1], frame.shape[0])  # Ret indica si la lectura fue exitosa (True) o no (False)
        cv2.imshow('Frame', frame_l)  # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  # Corto la repoducción si se presiona la tecla "q"
    else:
        break  # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.

cap.release()  # Cierro el video
cv2.destroyAllWindows()