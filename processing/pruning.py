import cv2
import numpy as np
from esqueleta import esqueleta

def bridge(image):
    bridged = image.copy()
    rows, cols = image.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == 0:  # Se for pixel de fundo
                # Verifica padrões de pixels diagonais que podem ser "pontes"
                if (image[i-1, j-1] == 255 and image[i+1, j+1] == 255) or \
                   (image[i-1, j+1] == 255 and image[i+1, j-1] == 255):
                    bridged[i, j] = 255
    return bridged

def pruning(bw):
    # Elimina pequenos 'ramos'
    kernel = np.ones((3, 3), np.uint8)
    bwo = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=-1)

    # Substitui MORPH_BRIDGE pela nossa função bridge
    bwbrid = bridge(bwo)

    # Continua com o resto do processamento
    bwclean = cv2.morphologyEx(bwbrid, cv2.MORPH_OPEN, kernel, iterations=-1)
    bwfill = cv2.morphologyEx(bwclean, cv2.MORPH_CLOSE, kernel, iterations=-1)

    # Chama Esqueletiza duas vezes
    bw2 = esqueleta(bwfill)
    bw2 = esqueleta(bw2)

    return bw2
