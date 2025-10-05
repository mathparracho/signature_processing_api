import numpy as np
from skimage.transform import resize

def resize_images_to_max(img1, img2):
    shapes = [img.shape for img in [img1, img2]]
    max_h = max(shape[0] for shape in shapes)
    max_w = max(shape[1] for shape in shapes)
    
    def resize_binary(img):
        resized = resize(img, (max_h, max_w), order=0, preserve_range=True, anti_aliasing=False)
        return np.where(resized > 127, 255, 0).astype(np.uint8)
    
    return resize_binary(img1), resize_binary(img2)
