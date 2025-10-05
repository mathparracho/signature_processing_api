import numpy as np

def esqueleta(bw_image):
    """Implementação do algoritmo Zhang-Suen para esqueletização"""
   
    if bw_image.dtype != np.uint8:
        bw_image = bw_image.astype(np.uint8)
    

    if np.max(bw_image) == 255:
        bw_image = (bw_image / 255).astype(np.uint8)
    
    def neighbours(x, y, image):
        """Return 8-neighbours of image point P1(x,y), clockwise"""
        return [
            image[x-1][y],   # P2
            image[x-1][y+1], # P3
            image[x][y+1],    # P4
            image[x+1][y+1],  # P5
            image[x+1][y],    # P6
            image[x+1][y-1],  # P7
            image[x][y-1],    # P8
            image[x-1][y-1]   # P9
        ]
    
    def transitions(neighbours):
        """No. of 0,1 patterns (transitions from 0 to 1)"""
        n = neighbours + neighbours[0:1]  # P2, P3, ..., P9, P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))
    
    Image_Thinned = bw_image.copy()
    changing1 = changing2 = 1
    
    while changing1 or changing2:
        
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and     
                    2 <= sum(n) <= 6 and             
                    transitions(n) == 1 and          
                    P2 * P4 * P6 == 0 and           
                    P4 * P6 * P8 == 0):             
                    changing1.append((x, y))
        
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        
        
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and     
                    2 <= sum(n) <= 6 and             
                    transitions(n) == 1 and          
                    P2 * P4 * P8 == 0 and           
                    P2 * P6 * P8 == 0):             
                    changing2.append((x, y))
        
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    
   
    if np.max(Image_Thinned) == 1:
        Image_Thinned = Image_Thinned * 255
    
    return Image_Thinned.astype(np.uint8)
