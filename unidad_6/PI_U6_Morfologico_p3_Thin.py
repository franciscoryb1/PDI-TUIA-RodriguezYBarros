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

# -------------------------------------------------------------------------------------------------------------
# --- Thining -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool_)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool_)

def imthin(image, n_iter=None):
    """
    imthin: Realiza el adelgazamiento (thinning) morfológico de una imagen binaria.

    Parametros
    ----------
    image   : Imagen binaria (M,N) ndarray, valores permitidos: 0 y 1.
    n_iter  : int, numero de iteraciones (opcional).

    Salida
    ------
    out     : Imagen binaria (M,N) ndarray. Imagen adelgazada (thinned).

    Referencias
    ----------
    [1] Z. Guo and R. W. Hall, "Parallel thinning with two-subiteration algorithms," Comm. ACM, vol. 32, no. 3, pp. 359-373, 1989.
    [2] Lam, L., Seong-Whan Lee, and Ching Y. Suen, "Thinning Methodologies-A Comprehensive Survey," IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol 14, No. 9, 1992, p. 879
    """
    # --- Check n_iter -----------------------------------------
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # --- Check image ------------------------------------------
    skel = np.array(image).astype(np.uint8)
    if skel.ndim != 2:
        raise ValueError('"image" debe ser un array 2D.')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('"image" debe contener valores 0s y 1s solamente.')

    # --- Definiciones iniciales -------------------------------
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # --- Procesamiento ----------------------------------------
    while n != 0:
        before = np.sum(skel)                                                   # Cuento la cantidad de pixels True antes de esta iteración...
        # ----------------------------------
        for lut in [G123_LUT, G123P_LUT]:                                       # sub-itero para cada LUT
            N = cv2.filter2D(skel, -1, mask, borderType=cv2.BORDER_CONSTANT)    # Correlación entre la imagen y la máscara 
            D = np.take(lut, N)                                                 # Decido que pixels eliminar en base al resultado de la correlación y la LUT.
            skel[D] = 0                                                         # Elimino pixels (adelgazo)            
        # ----------------------------------
        after = np.sum(skel)                                                    # Cuento la cantidad de pixels True al final de esta iteración.
        if before == after:                                                     # Si no hay cambios, no itero mas, termine el adelgazamiento...
            break   
        n -= 1                                                                  # Caso contrario, decremento n
    return skel.astype(np.bool_)                                                
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# --- Cargo Imagen --------------------------------------------------
f = cv2.imread('cuadrado.tif', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('O.png', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('opencv.png', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('GT.tif', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('fingerprint_cleaned.tif', cv2.IMREAD_GRAYSCALE)
np.unique(f)

# --- Acondicionamiento ---------------------------------------------
th, fbw = cv2.threshold(f,127, 1, cv2.THRESH_OTSU)
np.unique(fbw)

# --- Thin ----------------------------------------------------------
f_thin = imthin(fbw)

plt.figure()
ax1 = plt.subplot(121); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(f_thin, cmap='gray'), plt.title('Thin'), plt.xticks([]), plt.yticks([])
plt.show(block=False)


