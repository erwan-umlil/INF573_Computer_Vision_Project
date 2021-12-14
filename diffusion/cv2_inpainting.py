import numpy as np
import cv2 as cv2 # opencv computer vision library
from skimage import io # for io.imread



img = cv2.imread( 'diffusion/img/s_bird.png') 
mask = cv2.imread('diffusion/img/s_bird_mask.png', 0)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

output = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('image with white holes', img)
cv2.imshow('image post inpainting cv2', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
