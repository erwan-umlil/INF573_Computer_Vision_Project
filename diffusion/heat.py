from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


K = 5e-1
dx = 1.0
dy = 1.0

img = Image.open('diffusion/s_bird.png').convert('RGB')
img = np.array(img)
mask = Image.open('diffusion/s_bird_mask.png').convert('L')
mask = np.array(mask).astype(float)
mask /= np.max(mask)

n, m, _ = img.shape
res = img.copy() - 255 * mask[:,:,np.newaxis].astype(int)

for epoch in tqdm(range(100)):
    for i in range(1, n-1):
        for j in range(1, m-1):
            if mask[i,j] < 0.5: # not in a hole
                continue
            diff = K * ((res[i-1,j] - 2*res[i,j] + res[i+1,j]) / (dx**2) + (res[i,j-1] - 2*res[i,j] + res[i,j+1]) / (dy**2))
            res[i,j] += diff.astype(int)

plt.imshow(res)
plt.show()
