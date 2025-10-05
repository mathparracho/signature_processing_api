import cv2
import numpy as np
from math import atan2, degrees, radians, sin, cos

def alinhar_imagem_com_centroides(imagem_original, centroide_total, centroide_direita, centroide_esquerda):
    """
    Alinha a imagem com base na inclinação da linha formada pelos centróides.
    
    Parâmetros:
    imagem_original (numpy.ndarray): Imagem colorida ou em tons de cinza
    centroide_total, centroide_direita, centroide_esquerda: Tuplas (x,y) com as coordenadas dos centróides
    
    Retorna:
    numpy.ndarray: Imagem rotacionada e alinhada
    float: Ângulo de correção aplicado (em graus)
    """

    centroides_validos = [c for c in [centroide_total, centroide_direita, centroide_esquerda] if c is not None]
    
    if len(centroides_validos) < 2:
        return imagem_original, 0.0  
    
   
    pontos = np.array(centroides_validos, dtype=np.float32)
    

    # y = a*x + b
    x = pontos[:, 0]
    y = pontos[:, 1]
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x2 = np.sum(x*x)
    
    denominador = n * sum_x2 - sum_x**2
    if denominador == 0:
        return imagem_original, 0.0  
    
    a = (n * sum_xy - sum_x * sum_y) / denominador
    b = (sum_y - a * sum_x) / n
    
    angle = degrees(atan2(a, 1))  
    altura, largura = imagem_original.shape[:2]
    theta = radians(angle)
    
    new_width = int(abs(largura * cos(theta)) + abs(altura * sin(theta)))
    new_height = int(abs(largura * sin(theta)) + abs(altura * cos(theta)))
 
    matriz_rot = cv2.getRotationMatrix2D((largura/2, altura/2), angle, 1.0)
    
    # Ajustar a translação na matriz de rotação para manter a imagem centralizada
    matriz_rot[0, 2] += (new_width - largura) / 2
    matriz_rot[1, 2] += (new_height - altura) / 2
    
    # Aplicar a transformação afim
    imagem_alinhada = cv2.warpAffine(imagem_original, matriz_rot, (new_width, new_height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    
    return imagem_alinhada, angle, theta
