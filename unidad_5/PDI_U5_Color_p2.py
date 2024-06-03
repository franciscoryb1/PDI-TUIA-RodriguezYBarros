import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   # Pillow --> https://pillow.readthedocs.io/en/stable/

# --------------------------------------------------------------------------------------
# --- Dithering ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# --- Imagen en escala de grises - Ejemplo 1 -------------------------------------------
img_cv2 = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
img_cv2.shape

img_PIL = Image.open('cameraman.tif')
img_PIL.mode    # L: Luminance 
img_PIL.size    # (256,256)

image_dithering = img_PIL.convert(mode='1', dither=Image.FLOYDSTEINBERG)   # https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=convert#PIL.Image.Image.convert
plt.figure()
ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(img_PIL, cmap='gray'), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show(block=False)

# -- Análisis Imagen Original -------------------
img_cv2.shape
np.unique(img_cv2)
len(np.unique(img_cv2))                     # 247 = 253-7+1

img_PIL.size
col_and_counts = img_PIL.getcolors()        # [ ( count, index ), ( count, index ), ... ]
len(col_and_counts)

x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)                         # 256*256 = 65.536
list_of_pixels[:5]
np.unique(list_of_pixels)
Ncolors = len(np.unique(list_of_pixels))    # 247 = 253-7+1

# -- Análisis Imagen Procesada ------------------
col_and_counts = image_dithering.getcolors()    # [ ( count, index ), ( count, index ), ... ]
len(col_and_counts)

x = image_dithering.getdata()
list_of_pixels = list(x)
len(list_of_pixels)                             # 256*256 = 65.536
list_of_pixels[:5]
np.unique(list_of_pixels)
Ncolors_out = len(np.unique(list_of_pixels))    # 2

# Obtengo matriz de datos
image_dithering_data = np.array(image_dithering.getdata(), dtype=np.uint8).reshape(img_PIL.size)
type(image_dithering_data)
image_dithering_data.dtype
plt.figure(), plt.imshow(image_dithering_data, cmap='gray'), plt.show(block=False)


# --- Imagen Color - Ejemplo 1 ---------------------------------------------------------
img_PIL = Image.open('landscape.jpg')
img_PIL.size
img_PIL.mode
img_proc = img_PIL.convert(mode="P", dither=Image.NONE, palette=Image.WEB)                      # standard 216-color "web palette"
img_proc_dither = img_PIL.convert(mode="P", dither=Image.FLOYDSTEINBERG, palette=Image.WEB)     # standard 216-color "web palette"
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img_PIL), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_proc), plt.title('Imagen procesada')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_proc_dither), plt.title('Imagen procesada + dither')
plt.show(block=False)

# -- Análisis Imagen Original -------------------
img_PIL.size                # 720 x 480 = 345.600
col_and_counts = img_PIL.getcolors(img_PIL.size[0]*img_PIL.size[1])     # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getcolors
len(col_and_counts)

x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)         # 720 x 480 = 345.600       
list_of_pixels[:5]

# -- Análisis img_proc --------------------------
col_and_counts = img_proc.getcolors(img_proc.size[0]*img_proc.size[1])    
len(col_and_counts)

x = img_proc.getdata()
list_of_pixels = list(x)
len(list_of_pixels)                             
list_of_pixels[:5]
np.unique(list_of_pixels)
Ncolors_out = len(np.unique(list_of_pixels))  
img_proc_idxs = np.array(list_of_pixels, dtype=np.uint8).reshape(img_proc.size[::-1])   # Tener cuidado con el reshape...
img_proc_idxs.shape
plt.figure()
ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(img_proc), plt.title('Imagen procesada - RGB')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(img_proc_idxs, cmap='gray'), plt.title('Imagen procesada - indices')
plt.show(block=False)

paleta = img_proc.getpalette()      # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getpalette
type(paleta)
len(paleta)                         # 768 = 256*3
paleta = np.reshape(paleta, (-1,3))
paleta.max()
paleta.min()
paleta.dtype
paleta = paleta.astype(np.uint8)
aux = np.array([c for c in paleta if c.any()])  # Acá se puede ver que hay 216 colores --> standard 216-color "web palette":
aux.shape                                       #       215 x 3  + [0 0 0] = 216 x 3

img_proc_RGB = np.array(img_proc.convert('RGB'))  # Paso a RGB
plt.figure(), plt.imshow(img_proc_RGB), plt.show(block=False)
colours = np.unique(img_proc_RGB.reshape(-1,3), axis=0)
colours.shape

# -- Análisis img_proc_dither --------------------------
dither_col_and_counts = img_proc_dither.getcolors(img_proc_dither.size[0]*img_proc_dither.size[1])    
len(dither_col_and_counts)

x = img_proc_dither.getdata()
list_of_pixels = list(x)
np.unique(list_of_pixels)
dither_Ncolors_out = len(np.unique(list_of_pixels))  
dither_img_proc_idxs = np.array(list_of_pixels, dtype=np.uint8).reshape(img_proc_dither.size[::-1])   # Tener cuidado con el reshape...
dither_img_proc_idxs.shape
plt.figure()
ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(img_proc_dither), plt.title('Imagen procesada - RGB')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(dither_img_proc_idxs, cmap='gray'), plt.title('Imagen procesada - indices')
plt.show(block=False)

dither_paleta = np.reshape(img_proc_dither.getpalette(), (-1,3)).astype(np.uint8)
dither_paleta.shape
np.all(dither_paleta==paleta)   # Las dos paletas son iguales...

img_proc_dither_RGB = np.array(img_proc_dither.convert('RGB'))  # Paso a RGB
plt.figure(), plt.imshow(img_proc_dither_RGB), plt.show(block=False)
dither_colours = np.unique(img_proc_dither_RGB.reshape(-1,3), axis=0)
dither_colours.shape


# --- Imagen Color - Ejemplo 2 ---------------------------------------------------------
img_PIL = Image.open('peppers.png')
# image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
# image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG, colors=3)
image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG, colors=8)
plt.figure()
ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(img_PIL), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show(block=False)

# -- Análisis Imagen Original -------------------
img_PIL.size
x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)
Ncolors = len(list(set(list_of_pixels)))

# -- Análisis Imagen Procesada -------------------
list_of_pixels_out = list(image_dithering.getdata())
np.unique(list_of_pixels_out)
Ncolors_out = len(np.unique(list_of_pixels_out))

image_dithering.getcolors() # [ ( count, index ), ( count, index ), ... ]
palette = np.array(image_dithering.getpalette(),dtype=np.uint8).reshape((-1,3))
palette[0:9,]

# Paso a RGB
image_dithering_RGB = np.array(image_dithering.convert('RGB'))  # Paso a RGB
colours, counts = np.unique(image_dithering_RGB.reshape(-1,3), axis=0, return_counts=1)    # Obtengo colores y cuentas

# Grafico de torta --> Tener cuidado con la cantidad de colores! debe ser relativamente chica para hacer este grafico
idx = np.argsort(-counts)   # Esto es opcional...
counts = counts[idx]        # Ordeno en base a frecuencia de ocurrencias 
colours = colours[idx]      # 
counts_pct = counts/np.sum(counts)*100  # Paso a porcentaje
# labels = [f'({c[0]},{c[1]},{c[2]})' for c in colours]
# labels = [f'{counts_pct[ii]:5.2f}%  ({colours[ii,0]},{colours[ii,1]},{colours[ii,2]})' for ii in range(len(counts))]
labels = [f'{counts_pct[ii]:6.2f}%  ({colours[ii,0]:3d},{colours[ii,1]:3d},{colours[ii,2]:3d})' for ii in range(len(counts))]
col = [(c[0]/255., c[1]/255., c[2]/255.) for c in colours]

plt.figure(figsize=(9,5))
ax = plt.subplot(111) 
ax.pie(counts_pct, labels=labels, colors=col)
pos1 = ax.get_position()
pos2 = [0.15, pos1.y0, pos1.width, pos1.height] 
ax.set_position(pos2) 
plt.legend(title = "Colores RGB", bbox_to_anchor=(1.4, 1), loc='upper left', fontsize=10)   # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
plt.title("Proporción de Colores")
plt.show(block=False)

# Obtengo índices
image_dithering_idxs = np.array(list(image_dithering.getdata()), dtype=np.uint8).reshape(image_dithering.size[::-1])   # Tener cuidado con el reshape...
image_dithering_idxs.shape
np.unique(image_dithering_idxs)
plt.figure()
ax1 = plt.subplot(121); plt.xticks([]), plt.yticks([]), plt.imshow(image_dithering), plt.title('Imagen procesada - RGB')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering_idxs, cmap='gray'), plt.title('Imagen procesada - indices')
plt.show(block=False)
