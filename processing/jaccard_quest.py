import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def plot_jaccard_local_fast(matriz1, matriz2, raio_vizinhanca):
    
    if matriz1.shape != matriz2.shape:
        raise ValueError('Matrizes devem ter mesmas dimensões')
    
    m1 = (matriz1 > 0).astype(float)
    m2 = (matriz2 > 0).astype(float)

    kernel_size = 2 * raio_vizinhanca + 1
    kernel = np.ones((kernel_size, kernel_size))
    
    # convolução
    intersecao = convolve2d(m1 * m2, kernel, mode='same')  
    uniao = convolve2d(np.maximum(m1, m2), kernel, mode='same') 
    
    jaccard_local = np.divide(intersecao, uniao, 
                             out=np.ones_like(intersecao),
                             where=(uniao != 0))       
    
    # máscara de borda
    mask = np.zeros_like(m1)
    mask[raio_vizinhanca:-raio_vizinhanca, 
         raio_vizinhanca:-raio_vizinhanca] = 1
    jaccard_local[mask == 0] = np.nan
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(matriz1)
    plt.title('Paradigmática 1')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(matriz2)
    plt.title('Questionada')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    img = plt.imshow(jaccard_local, vmin=0, vmax=1)
    plt.title(f'Índice de Jaccard Local (Raio={raio_vizinhanca})')
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    jaccard_global = np.sum(matriz1 & matriz2) / np.sum(matriz1 | matriz2)
    plt.suptitle(f'Jaccard Global = {jaccard_global:.3f}')
    
    plt.tight_layout()
    plt.savefig("jaccard.png")
