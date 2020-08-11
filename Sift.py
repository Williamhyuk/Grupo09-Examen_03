import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('taza1.jpeg',cv.IMREAD_GRAYSCALE)         
img2 = cv.imread('taza2.jpeg',cv.IMREAD_GRAYSCALE) 
# Iicialmizamos detector ORB
orb = cv.ORB_create()
#buscamos puntos clave y descriptores con ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# creamos objetos BFMatcher 
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match
matches = bf.match(des1,des2)
# Ordenamos  según la  distancia.
matches = sorted(matches, key = lambda x:x.distance)
# Dibujamos los 30 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
