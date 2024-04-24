import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Cargo imagen y pre-proceso ------------------------------------------------------------
img = cv2.imread("faces.jpg")                     # Cargo imagen
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
plt.figure(), plt.imshow(img), plt.show(block=False)

# --- ROIS ----------------------------------------------------------------------------------
face_1 = [116,  60,  55,  55] # [x, y, w, h]
face_2 = [245,  21,  56,  56] 
face_3 = [377,  42,  50,  50]
face_4 = [468,  77,  54,  54]

img_faces = img.copy()        
cv2.rectangle(img_faces, (face_1[0], face_1[1]), (face_1[0]+face_1[2], face_1[1]+face_1[3]), (255,0,0), 2)
cv2.rectangle(img_faces, (face_2[0], face_2[1]), (face_2[0]+face_2[2], face_2[1]+face_2[3]), (255,0,0), 2)
cv2.rectangle(img_faces, (face_3[0], face_3[1]), (face_3[0]+face_3[2], face_3[1]+face_3[3]), (255,0,0), 2)
cv2.rectangle(img_faces, (face_4[0], face_4[1]), (face_4[0]+face_4[2], face_4[1]+face_4[3]), (255,0,0), 2)
plt.figure(), plt.imshow(img_faces), plt.show(block=False)

roi_1 = img[face_1[1]:face_1[1]+face_1[3], face_1[0]:face_1[0]+face_1[2]]
roi_2 = img[face_2[1]:face_2[1]+face_2[3], face_2[0]:face_2[0]+face_2[2]]
roi_3 = img[face_3[1]:face_3[1]+face_3[3], face_3[0]:face_3[0]+face_3[2]]
roi_4 = img[face_4[1]:face_4[1]+face_4[3], face_4[0]:face_4[0]+face_4[2]]
plt.figure()
plt.subplot(221), plt.imshow(roi_1)
plt.subplot(222), plt.imshow(roi_2)
plt.subplot(223), plt.imshow(roi_3)
plt.subplot(224), plt.imshow(roi_4)
plt.show(block=False)

# --- Borrosidad ----------------------------------------------------------------------------
K = 15
roi_1_blurred = cv2.blur(roi_1, (K,K))
roi_2_blurred = cv2.blur(roi_2, (K,K))
roi_3_blurred = cv2.blur(roi_3, (K,K))
roi_4_blurred = cv2.blur(roi_4, (K,K))
plt.figure(), 
plt.subplot(221), plt.imshow(roi_1_blurred)
plt.subplot(222), plt.imshow(roi_2_blurred)
plt.subplot(223), plt.imshow(roi_3_blurred)
plt.subplot(224), plt.imshow(roi_4_blurred)
plt.show(block=False)

# --- Imagen con rostros difuminados --------------------------------------------------------
img_faces_blurred = img.copy()  
img_faces_blurred[face_1[1]:face_1[1]+face_1[3], face_1[0]:face_1[0]+face_1[2]] = roi_1_blurred
img_faces_blurred[face_2[1]:face_2[1]+face_2[3], face_2[0]:face_2[0]+face_2[2]] = roi_2_blurred
img_faces_blurred[face_3[1]:face_3[1]+face_3[3], face_3[0]:face_3[0]+face_3[2]] = roi_3_blurred
img_faces_blurred[face_4[1]:face_4[1]+face_4[3], face_4[0]:face_4[0]+face_4[2]] = roi_4_blurred
plt.figure(), plt.imshow(img_faces_blurred), plt.show(block=False)
