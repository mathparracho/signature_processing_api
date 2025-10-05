import numpy as np

def esqueleta(bw_image):
    # Esqueletiza a imagem binária
    continue_it = True
    while continue_it:
        BnW_old = bw_image.copy()
        BaW_del = np.zeros_like(bw_image, dtype=np.uint8)

        # Primeira iteração
        for i in range(1, bw_image.shape[0] - 1):
            for j in range(1, bw_image.shape[1] - 1):
                P = [
                    bw_image[i, j],
                    bw_image[i-1, j],
                    bw_image[i-1, j+1],
                    bw_image[i, j+1],
                    bw_image[i+1, j+1],
                    bw_image[i+1, j],
                    bw_image[i+1, j-1],
                    bw_image[i, j-1],
                    bw_image[i-1, j-1],
                    bw_image[i-1, j]
                ]
                
                if P[1] * P[3] * P[5] == 0 and P[3] * P[5] * P[7] == 0 and 2 <= sum(P[1:9]) <= 6:
                    A = 0
                    for k in range(1, len(P)-1):
                        if P[k] == 0 and P[k+1] == 1:
                            A += 1
                    
                    if A == 1:
                        BaW_del[i, j] = 1

        bw_image[BaW_del == 1] = 0

        # Segunda iteração
        BaW_del = np.zeros_like(bw_image, dtype=np.uint8)
        for i in range(1, bw_image.shape[0] - 1):
            for j in range(1, bw_image.shape[1] - 1):
                P = [
                    bw_image[i, j],
                    bw_image[i-1, j],
                    bw_image[i-1, j+1],
                    bw_image[i, j+1],
                    bw_image[i+1, j+1],
                    bw_image[i+1, j],
                    bw_image[i+1, j-1],
                    bw_image[i, j-1],
                    bw_image[i-1, j-1],
                    bw_image[i-1, j]
                ]
                
                if P[1] * P[3] * P[7] == 0 and P[1] * P[5] * P[7] == 0 and 2 <= sum(P[1:9]) <= 6:
                    A = 0
                    for k in range(1, len(P)-1):
                        if P[k] == 0 and P[k+1] == 1:
                            A += 1
                    
                    if A == 1:
                        BaW_del[i, j] = 1

        bw_image[BaW_del == 1] = 0
        
        # Verificar se houve mudanças
        if np.all(BnW_old == bw_image):
            continue_it = False

    return bw_image
