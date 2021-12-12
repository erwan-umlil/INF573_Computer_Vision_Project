from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm

img = Image.open('diffusion/s_bird.png').convert('RGB')
img = np.array(img)
mask = Image.open('diffusion/s_bird_mask.png').convert('L')
mask = np.array(mask).astype(float)
mask /= np.max(mask)

n, m, _ = img.shape
res = img.copy() - 255 * mask[:,:,np.newaxis].astype(int)
state = (np.ones_like(mask) - mask).astype(int)
for epoch in tqdm(range(100)):
    if epoch%10 == 0:
        state = (np.ones_like(mask) - mask).astype(int)
    for i in range(1, n-1):
        for j in range(1, m-1):
            if mask[i,j] < 0.5: # not in a hole
                continue
            window = state[i-1:i+2, j-1:j+2] * res[i-1:i+2, j-1:j+2]
            res[i,j] = np.sum(window)/np.sum(window != 0)
            state[i,j] = 1

plt.imshow(res)
plt.show()
