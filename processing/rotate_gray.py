import numpy as np
import cv2
from math import cos, sin

def rotate_gray(imagem_original, angle, theta):

 altura, largura = imagem_original.shape[:2]
 new_width = int(abs(largura * cos(theta)) + abs(altura * sin(theta)))
 new_height = int(abs(largura * sin(theta)) + abs(altura * cos(theta)))
 
 matriz_rot = cv2.getRotationMatrix2D((largura/2, altura/2), angle, 1.0)
    
    # Ajustar a translação na matriz de rotação para manter a imagem centralizada
 matriz_rot[0, 2] += (new_width - largura) / 2
 matriz_rot[1, 2] += (new_height - altura) / 2
    
    # Aplicar a transformação afim
 imagem_alinhada = cv2.warpAffine(imagem_original, matriz_rot, (new_width, new_height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=255)

 return imagem_alinhada
    
