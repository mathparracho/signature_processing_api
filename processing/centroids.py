import cv2
import numpy as np

def centroids(imagem_binaria):
    """
    Detecta os centróides de uma imagem binária:
    - Centróide da imagem inteira
    - Centróide da metade direita
    - Centróide da metade esquerda
    
    Parâmetros:
    imagem_binaria (numpy.ndarray): Imagem binária (0 e 255) onde 255 é o objeto de interesse
    
    Retorna:
    tuple: (centroide_total, centroide_direita, centroide_esquerda)
            Cada centróide é uma tupla (cx, cy) ou None se não encontrado
    """
    if len(imagem_binaria.shape) > 2:
        raise ValueError("A imagem deve ser binária (1 canal)")
    momentos = cv2.moments(imagem_binaria)
    if momentos["m00"] != 0:
        cx_total = int(momentos["m10"] / momentos["m00"])
        cy_total = int(momentos["m01"] / momentos["m00"])
        centroide_total = (cx_total, cy_total)
    else:
        centroide_total = None
    
    altura, largura = imagem_binaria.shape
    metade_direita = imagem_binaria[:, largura//2:]
    momentos_direita = cv2.moments(metade_direita)
    
    if momentos_direita["m00"] != 0:
        cx_d = int(momentos_direita["m10"] / momentos_direita["m00"]) + largura//2
        cy_d = int(momentos_direita["m01"] / momentos_direita["m00"])
        centroide_direita = (cx_d, cy_d)
    else:
        centroide_direita = None

    metade_esquerda = imagem_binaria[:, :largura//2]
    momentos_esquerda = cv2.moments(metade_esquerda)
    if momentos_esquerda["m00"] != 0:
        cx_e = int(momentos_esquerda["m10"] / momentos_esquerda["m00"])
        cy_e = int(momentos_esquerda["m01"] / momentos_esquerda["m00"])
        centroide_esquerda = (cx_e, cy_e)
    else:
        centroide_esquerda = None
    
    return centroide_total, centroide_direita, centroide_esquerda
