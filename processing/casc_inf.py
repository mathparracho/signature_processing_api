import numpy as np

def cas_inf(imagem_binaria):

    imagem_binaria = np.flipud(imagem_binaria)

    altura, largura = imagem_binaria.shape
    cas = np.zeros(largura, dtype=int)
    imagem_modificada = imagem_binaria.copy()  
    
    for x in range(largura):
        for y in range(altura):  # Varre de cima para baixo
            if imagem_modificada[y, x] == 255:
                cas[x] = y
                imagem_modificada[y, x] = 0  # Remove o pixel encontrado
                break  # Encontrou o primeiro pixel, passa para a próxima coluna


    imagem_modificada = np.flipud(imagem_modificada)
    
    return cas, imagem_modificada
