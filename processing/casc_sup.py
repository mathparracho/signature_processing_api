import numpy as np

def cas_sup(imagem_binaria):

    imagem_binaria = np.flipud(imagem_binaria)
    altura, largura = imagem_binaria.shape
    cas = np.zeros(largura, dtype=int)
    imagem_modificada = imagem_binaria.copy()
    
    for x in range(largura):
        for y in range(altura-1, -1, -1):  # Varre de baixo para cima
            if imagem_modificada[y, x] == 255:
                cas[x] = y
                imagem_modificada[y, x] = 0  # Remove o pixel encontrado
                break  

    imagem_modificada = np.flipud(imagem_modificada)
    return cas, imagem_modificada
