import cv2
import matplotlib.pyplot as plt

# Cargamos la imagen
img = cv2.imread('TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

def ecualizacion_local_histograma(img, ventana):

    alto, ancho = img.shape

    # buscamos el punto medio de nuestra ventana
    centro = ventana // 2

    # Creamos una copia de la imagen para no modificar la imagen original
    img_salida = img.copy()

    #iteramos sobre toda la imagen
    for y in range(centro, alto - centro):
        for x in range(centro, ancho - centro):

            # Extraemos la ventana local (ponemos +1 porque no incluye el último sino)
            ventana = img[y - centro:y + centro + 1, x - centro:x + centro + 1]

            # Calculamos el histograma local
            hist = cv2.calcHist([ventana], [0], None, [256], [0, 256])

            # Aplicamos la ecualización del histograma local
            ventana_ecualizada = cv2.equalizeHist(ventana)

            # Colocamos los píxeles ecualizados en la imagen de salida
            img_salida[y, x] = ventana_ecualizada[centro , centro]

    return img_salida


# Aplicamos la ecualización local del histograma a distintos tamaños de ventana

plt.subplot(221)
h = plt.imshow(img, cmap='gray')
plt.title('Imagen original')

#en esta ventana vemos demasiado ruido
plt.subplot(222)
img_salida1 = ecualizacion_local_histograma(img,10)
plt.imshow(img_salida1, cmap='gray')
plt.title('Imagen ventana 10')

#esta ventana nos muestra bastante bien los detalles escondidos en la imagen
plt.subplot(223)
img_salida2 = ecualizacion_local_histograma(img,30)
plt.imshow(img_salida2, cmap='gray')
plt.title('Imagen ventana 30')

#esta ventana ya empieza a perder los detalles pq estamos tomando demasaida información del entorno
plt.subplot(224)
img_salida3 = ecualizacion_local_histograma(img,50)
plt.imshow(img_salida3, cmap='gray')
plt.title('Imagen ventana 50')
plt.show()

#Lo que se puede ver en la imagen es:
#en el cuadrante superior izquierdo un cuadrado en el centro de la imagen
#en el cuadrante superior derecho una línea recta en diagonal
#en el centro se dibuja una letra a
#en el cuadrante inferior izquierdo 4 líneas horizontales
#en el cuadrante inferior derecho un círculo