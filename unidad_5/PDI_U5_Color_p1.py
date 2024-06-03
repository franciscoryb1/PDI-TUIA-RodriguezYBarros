import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Imagen RGB ------------------------------------------------------------------------
img = cv2.imread('peppers.png')
plt.figure(1), plt.imshow(img), plt.show(block=False)  # Acá se puede observar que OpenCV carga la imagen como BGR.

# --- Acomodamos canales ----------------------------------------------------------------
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(2), plt.imshow(img_RGB), plt.show(block=False)

# --- Separar canales -------------------------------------------------------------------
B, G, R = cv2.split(img)
plt.figure(3), plt.imshow(R, cmap='gray'), plt.title("Canal R"), plt.colorbar(), plt.show(block=False)

plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(R,cmap='gray'), plt.title('Canal R')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(G,cmap='gray'), plt.title('Canal G')
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(B,cmap='gray'), plt.title('Canal B')
plt.show(block=False)

# --- Modifico un canal ---------------------------------------------------------------
# img2 = img_RGB    # No! así crea una referencia: si se modifica una, se modifica la otra también.  
img2 = img_RGB.copy()  # Así crea una copia. Otra forma sería "img2 = np.array(img_RGB)"
img2[:,:,0] = 0
plt.figure, plt.imshow(img2), plt.title('Canal R anulado'), plt.show(block=False)

R2 = R.copy()
R2 = R2*0.5
R2 = R2.astype(np.uint8)
img3 = cv2.merge((R2,G,B))
plt.figure, plt.imshow(img3), plt.title('Canal R escalado'), plt.show(block=False)

plt.figure
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img2,cmap='gray'), plt.title('Canal R anulado')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img3), plt.title('Canal R escalado')
plt.show()

# --- Imagen Indexada -------------------------------------------------------------------
img = cv2.imread('peppers.png')         # N_colours= 99.059  --> Nbytes_img_idx/Nbytes_img: uint32 = 1.83  
# img = cv2.imread('home.jpg')            # N_colours= 51.711  --> Nbytes_img_idx/Nbytes_img: uint32 = 1.59 | uint16 = 0.93
# img = cv2.imread('flowers.tif')       # N_colours= 120.260 --> Nbytes_img_idx/Nbytes_img: uint32 = 1.667  # Cuidado! puede demorar mucho en correr...
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Obtengo colores
img_pixels = img.reshape(-1,3)
# colours = np.unique(img_pixels, axis=0)
colours, counts = np.unique(img_pixels, axis=0, return_counts=True)
idx = np.argsort(counts)        # Opcional:
counts = counts[idx]            # Ordeno los colores segun su frecuencia de aparición
colours = colours[idx]          # El ultimo elemento de counts posee el color con mayor frec. de aparición.
N_colours = colours.shape[0]

# Genero imagen indexada
img_idx = -np.ones(img.shape[:-1])
for ii in range(N_colours):
    # --- Version legible ---------------------------------------------------------------------
    col_sel = colours[ii]
    maskR = img[:,:,0] == col_sel[0]
    maskG = img[:,:,1] == col_sel[1]
    maskB = img[:,:,2] == col_sel[2]
    mask = maskR & maskG & maskB
    img_idx[mask] = ii
    # --- Version compacta -------------------------------------------------------------------
    # img_idx[(img[:,:,0] == colours[ii][0]) & (img[:,:,1] == colours[ii][1]) & (img[:,:,2] == colours[ii][2])] = ii
    # --- Otra version -------------------------------------------------------------------------
    # mask = cv2.inRange(img, colours[ii], colours[ii])  # https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    # img_idx[mask>0] = ii

plt.figure(), plt.imshow(img_idx, cmap="gray"), plt.show(block=False)
img_idx.max()
img_idx.min()

# Check & Conversion
np.any(img_idx==-1) # Verificamos que ningún pixel quedó sin asignar...
img_idx = np.uint32(img_idx)        # Rango: [0 2^32-1] --> [0 4.294.967.295]
# img_idx = np.uint16(img_idx)      # Rango: [0 2^16-1] --> [0 65.535]
img_idx.dtype

# Calculo de relación de bytes
Nbytes_img_idx = 4*np.prod(img_idx.shape) + np.prod(colours.shape)  # uint32: 4 bytes
# Nbytes_img_idx = 2*np.prod(img_idx.shape) + np.prod(colours.shape)  # uint16: 2 bytes
Nbytes_img = np.prod(img.shape)
Nbytes_img_idx/Nbytes_img

# Plots
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img), plt.title('Imagen Original'), plt.colorbar()
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="gray"), plt.title('Imagen Indexada - Indices'), plt.colorbar()
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="jet"), plt.title('Imagen Indexada - jet'), plt.colorbar()
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="hot"), plt.title('Imagen Indexada - hot'), plt.colorbar()
plt.show(block=False)

# Plot opcional (cuando los colores están ordenados)
P = 20   # 5 - 20 - 50 - 80 
img_idx_topValues_mask = cv2.inRange(np.int32(img_idx), (N_colours-1)*(100-P)/100, (N_colours-1))
img_idx_topValues = cv2.bitwise_and(img, img, mask= img_idx_topValues_mask)
# img_idx_topValues = img.copy()                              # Lo mismo que antes...   
# img_idx_topValues[img_idx_topValues_mask<255, :] = 0            # pero de manera "manual".
porc = 100* np.sum(img_idx_topValues_mask>0) / np.prod(img_idx_topValues_mask.shape)

plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img), plt.title('Imagen Original'), plt.colorbar()
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_idx_topValues_mask, cmap="gray"), plt.title(f'Imagen Indexada - {P:5.2f}% top values - mask'), plt.colorbar()
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_idx_topValues), plt.title(f'Imagen Indexada - top values (%{porc:5.2f})'), plt.colorbar()
plt.show(block=False)

