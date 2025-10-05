import cv2
import numpy as np
from skimage.measure import regionprops

# ✅ todos os imports locais agora usam ponto (.) para serem relativos ao pacote "processing"
from .esqueleta import esqueleta
from .pruning import pruning
from .centroids import centroids
from .rotate import alinhar_imagem_com_centroides
from .rotate_gray import rotate_gray
from .pressure import f_pressure_skel
from .resize_all import resize_images_to_max
from .casc_inf import cas_inf
from .casc_sup import cas_sup
from .residuos import residuos
from .plot_paradig import plotar_vetores_paradig
from .plot_quest import plotar_vetores_quest
from .jaccard_paradig import plot_jaccard_local_fast_paradig
from .jaccard_quest import plot_jaccard_local_fast

import matplotlib.pyplot as plt


def processing_pairs(img1_path, img3_path):

    I_originalImgA = cv2.imread(img1_path)
    if I_originalImgA is None:
        print("Erro: Imagem paradigmática 1 não encontrada!")
        exit()
    I_ImageA_OGG = cv2.cvtColor(I_originalImgA, cv2.COLOR_BGR2GRAY)
    _, I_ImageA_OG = cv2.threshold(I_ImageA_OGG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    I_ImageA_OG = np.logical_not(I_ImageA_OG).astype(np.uint8) * 255

    #I_originalImgB = cv2.imread(img2_path)
    #if I_originalImgB is None:
    #    print("Erro: Imagem paradigmática 2 não encontrada!")
    #    exit()
    #I_ImageB_OGG = cv2.cvtColor(I_originalImgB, cv2.COLOR_BGR2GRAY)
    #_, I_ImageB_OG = cv2.threshold(I_ImageB_OGG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #I_ImageB_OG = np.logical_not(I_ImageB_OG).astype(np.uint8) * 255

    I_originalImgC = cv2.imread(img3_path)
    if I_originalImgC is None:
        print("Erro: Imagem questionada não encontrada!")
        exit()
    I_Image_FG = cv2.cvtColor(I_originalImgC, cv2.COLOR_BGR2GRAY)
    _, I_Image_F = cv2.threshold(I_Image_FG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    I_Image_F = np.logical_not(I_Image_F).astype(np.uint8) * 255

    I_bwImgA = pruning(I_ImageA_OG)
    #I_bwImgB = pruning(I_ImageB_OG)
    I_bwImgF = pruning(I_Image_F)

    def crop_to_bounding_box(bw_img, gray_img):
        props = regionprops(bw_img.astype(int))
        if len(props) == 0:
            return bw_img, gray_img
        box = props[0].bbox
        coord = (box[1], box[0], box[3] - box[1], box[2] - box[0])
        cropped_bw = bw_img[box[0]:box[2], box[1]:box[3]]
        cropped_gray = gray_img[box[0]:box[2], box[1]:box[3]]
        return cropped_bw, cropped_gray

    I_skeletonImgA, I_ImageA_OGG = crop_to_bounding_box(I_bwImgA, I_ImageA_OGG)
    #I_skeletonImgB, I_ImageB_OGG = crop_to_bounding_box(I_bwImgB, I_ImageB_OGG)
    I_skeletonImgF, I_Image_FG = crop_to_bounding_box(I_bwImgF, I_Image_FG)


    centroide_total_A, centroide_direita_A, centroide_esquerda_A = centroids(I_skeletonImgA)
    #centroide_total_B, centroide_direita_B, centroide_esquerda_B = centroids(I_skeletonImgB)
    centroide_total_F, centroide_direita_F, centroide_esquerda_F = centroids(I_skeletonImgF)

    img1_skel, angleA, thetaA = alinhar_imagem_com_centroides(I_skeletonImgA, centroide_total_A, centroide_direita_A, centroide_esquerda_A)
    #img2_skel, angleB, thetaB = alinhar_imagem_com_centroides(I_skeletonImgB, centroide_total_B, centroide_direita_B, centroide_esquerda_B)
    imgF_skel, angleF, thetaF = alinhar_imagem_com_centroides(I_skeletonImgF, centroide_total_F, centroide_direita_F, centroide_esquerda_F)

    I_ImageA_OGG = rotate_gray(I_ImageA_OGG, angleA, thetaA)
    #I_ImageB_OGG = rotate_gray(I_ImageB_OGG, angleB, thetaB)
    I_Image_FG = rotate_gray(I_Image_FG, angleF, thetaF)

    img1_skel, imgF_skel = resize_images_to_max(img1_skel, imgF_skel)
    I_ImageA_OGG, I_Image_FG = resize_images_to_max(I_ImageA_OGG, I_Image_FG)

    pressure_A = f_pressure_skel(img1_skel, I_ImageA_OGG)
    #pressure_B = f_pressure_skel(img2_skel, I_ImageB_OGG)
    pressure_F = f_pressure_skel(imgF_skel, I_Image_FG)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(
        np.arange(len(pressure_A)),
        pressure_A,
        color='blue',
        marker='o',
        s=10
    )
    plt.title('Pressão - Imagem 1')
    
    plt.subplot(1, 3, 2)
    #plt.scatter(
    #    np.arange(len(pressure_B)),
    #    pressure_B,
    #    color='blue',
    #    marker='o',
    #    s=10
    #)
    #plt.title('Pressão - Imagem 2')
    
    #plt.subplot(1, 3, 3)
    plt.scatter(
        np.arange(len(pressure_F)),
        pressure_F,
        color='red',
        marker='o',
        s=10
    )
    plt.title('Pressão - Imagem Questionada')
    
    plt.tight_layout()
    plt.savefig('pressure_comparison.png')

    cas_sup_1_A, img1_res = cas_sup(img1_skel)
    #cas_sup_1_B, img2_res = cas_sup(img2_skel)
    cas_sup_1_F, imgF_res = cas_sup(imgF_skel)

    cas_inf_1_A, img1_res = cas_inf(img1_res)
    #cas_inf_1_B, img2_res = cas_inf(img2_res)
    cas_inf_1_F, imgF_res = cas_inf(imgF_res)

    cas_sup_2_A, img1_res = cas_sup(img1_res)
    #cas_sup_2_B, img2_res = cas_sup(img2_res)
    cas_sup_2_F, imgF_res = cas_sup(imgF_res)
    
    cas_inf_2_A, img1_res = cas_inf(img1_res)
    #cas_inf_2_B, img2_res = cas_inf(img2_res)
    cas_inf_2_F, imgF_res = cas_inf(imgF_res)

    res1 = residuos(img1_res)
    #res2 = residuos(img2_res)
    resF = residuos(imgF_res)

    res1_flat = np.concatenate([np.array(x).flatten() for x in res1])
    #res2_flat = np.concatenate([np.array(x).flatten() for x in res2])
    resF_flat = np.concatenate([np.array(x).flatten() for x in resF])

    casc_A = np.concatenate([cas_sup_1_A, cas_sup_2_A, res1_flat])
    #casc_B = np.concatenate([cas_sup_1_B, cas_sup_2_B, res2_flat])
    casc_F = np.concatenate([cas_sup_1_F, cas_sup_2_F, resF_flat])

    plt.subplot(1, 3, 1)
    plt.scatter(
        np.arange(len(casc_A)),
        casc_A,
        color='blue',
        marker='o',
        s=10
    )
    plt.title('Cascas - Imagem 1')
    
    plt.subplot(1, 3, 2)
    plt.scatter(
    #    np.arange(len(casc_B)),
    #    casc_B,
    #    color='blue',
    #    marker='o',
    #    s=10
    #)
    #plt.title('Cascas - Imagem 2')
    
    #plt.subplot(1, 3, 3)
    #plt.scatter(
        np.arange(len(casc_F)),
        casc_F,
        color='red',
        marker='o',
        s=10
    )
    plt.title('Cascas - Imagem Questionada')

    #plotar_vetores_paradig(casc_A,casc_B,"Comparação de Vetores","Paradigmática 1","Paradigmática 2")
    plotar_vetores_quest(casc_A,casc_F,"Comparação de Vetores","Paradigmática 1","Questionada")

    #plot_jaccard_local_fast_paradig(img1_skel,img2_skel,raio_vizinhanca=2)
    plot_jaccard_local_fast(img1_skel,imgF_skel,raio_vizinhanca=2)

    print("Processamento concluído com sucesso!")

if __name__ == "__main__":
    print("Escolha as imagens:")
    img1_path = input("Imagem paradigmática 1: ")
    img1_path = "original_1_1.png"
    #img2_path = input("Imagem paradigmática 2: ")
    img3_path = input("Imagem questionada: ")
    img3_path = "original_2_9.png"
    processing_pairs(img1_path, img3_path)
