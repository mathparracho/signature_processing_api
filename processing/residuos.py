import numpy as np

def residuos(imagem_binaria):

    imagem_binaria = np.flipud(imagem_binaria)
  
    altura, largura = imagem_binaria.shape
    res = []
    
    for x in range(largura):
        # Encontra todos os índices y onde o valor é 255 na coluna x
        posicoes_y = np.where(imagem_binaria[:, x] == 255)[0]
        
        # Se não encontrar nenhum pixel branco, adiciona array com 0
        if len(posicoes_y) == 0:
            res.append(np.array([0]))
        else:
            res.append(posicoes_y)
    
    return res
