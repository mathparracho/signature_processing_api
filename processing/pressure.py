import numpy as np

def f_pressure_skel(full_skel, gray_lvl):
    # Resize gray_lvl to match full_skel dimensions if they differ
    if full_skel.shape != gray_lvl.shape:
        from skimage.transform import resize
        gray_lvl = resize(gray_lvl, full_skel.shape, preserve_range=True).astype(np.uint8)
    
    # Initialize output array (1D, length = number of columns)
    f_pressure = np.zeros(full_skel.shape[1], dtype=np.uint8)
    
    # Loop through each column
    for j in range(full_skel.shape[1]):
        # Find the first row where skeleton exists (MATLAB's find(..., 1))
        rows = np.where(full_skel[:, j] == 255)[0]
        if rows.size > 0:
            f_pressure[j] = gray_lvl[rows[0], j]
    
    return f_pressure
